# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``particles extract``'s per-page reporting.

Reported from dogfood: a single 83-page PDF printed 83 ``Page N: 0 particles``
lines, burying the other four snapshots in the same run — and it flagged pages
that chunk-hash carry-forward had legitimately satisfied as
``⚠ zero yield``, i.e. reported the cache working as an anomaly.

Per-item detail is already assigned to ``--verbose``; this listing had
simply never been wired to it.
"""

from __future__ import annotations

from typing import Any

# --------------------------------------------------------------------------
# Per-page reporting volume + the false zero-yield flag
# --------------------------------------------------------------------------


class TestPageStatReporting:
    """An 83-page PDF printed 83 lines unconditionally, and flagged pages that
    carry-forward had legitimately satisfied as ``⚠ zero yield``."""

    @staticmethod
    def _stats(total: int, with_particles: int) -> list[Any]:
        from particles.extraction.general import PageStat

        return [
            PageStat(page_number=i + 1, candidate_count=(3 if i < with_particles else 0))
            for i in range(total)
        ]

    def _render(self, stats: list[Any], carried: list[str], *, verbose: bool) -> str:
        import typer
        from typer.testing import CliRunner

        from particles.api.cli._output import configure_output
        from particles.api.cli.extract import _echo_page_stats

        cli = typer.Typer()

        @cli.command()
        def show() -> None:
            configure_output(verbose=verbose)
            _echo_page_stats(stats, carried)

        return CliRunner().invoke(cli, []).output

    def test_default_aggregates_instead_of_listing_every_page(self) -> None:
        out = self._render(self._stats(83, 12), [], verbose=False)
        assert "12 of 83 pages produced particles" in out
        assert "Page   1:" not in out
        assert out.count("\n") <= 2  # one line, not 83

    def test_verbose_itemises(self) -> None:
        out = self._render(self._stats(5, 2), [], verbose=True)
        assert "Pages: 5" in out
        assert "Page   1:" in out
        assert "Page   5:" in out

    def test_zero_yield_flagged_without_carry_forward(self) -> None:
        """An empty page with no cache hit is a real signal — keep it."""
        out = self._render(self._stats(3, 1), [], verbose=True)
        assert "⚠ zero yield" in out

    def test_zero_yield_not_flagged_when_carried_forward(self) -> None:
        """The cache working is not an anomaly. This was the false warning."""
        out = self._render(self._stats(3, 1), ["p-1", "p-2"], verbose=True)
        assert "⚠ zero yield" not in out

    def test_aggregate_explains_carry_forward(self) -> None:
        out = self._render(self._stats(83, 12), ["p-1"], verbose=False)
        assert "the rest matched already-extracted content" in out

    def test_all_pages_productive_prints_nothing_by_default(self) -> None:
        """Nothing to report is reported as nothing."""
        assert self._render(self._stats(4, 4), [], verbose=False).strip() == ""

    def test_no_page_stats_prints_nothing(self) -> None:
        assert self._render([], [], verbose=True).strip() == ""


# --------------------------------------------------------------------------
# --snapshot-id on an older generation
# --------------------------------------------------------------------------


class TestOlderSnapshotRefusal:
    """Naming a snapshot older than a living source's current version needs ``--force``."""

    @staticmethod
    def _seed() -> tuple[str, str, str]:
        import asyncio
        from datetime import UTC, datetime

        from particles.db import session_scope
        from tests._observer_scope import generation

        async def seed() -> tuple[str, str, str]:
            async with session_scope() as session:
                entry, older = await generation(
                    session, "note.md", "day one", [], captured_at=datetime(2026, 6, 1, tzinfo=UTC)
                )
                _, newer = await generation(
                    session, "note.md", "day two", [], captured_at=datetime(2026, 6, 2, tzinfo=UTC)
                )
                await session.commit()
                return entry, older, newer

        return asyncio.run(seed())

    def test_an_older_snapshot_is_refused(self, cli_db: Any) -> None:
        from typer.testing import CliRunner

        from particles.api.cli import app

        entry, older, _ = self._seed()
        refused = CliRunner().invoke(app, ["extract", entry, "--snapshot-id", older])

        assert refused.exit_code == 1
        assert "older than entry" in refused.output and "--force" in refused.output

    def test_force_and_the_current_snapshot_resolve(self, cli_db: Any) -> None:
        import asyncio

        from particles.api.cli.extract import _resolve_local

        entry, older, newer = self._seed()

        assert asyncio.run(_resolve_local(entry, older, force=True)) == (entry, older)
        assert asyncio.run(_resolve_local(entry, newer)) == (entry, newer)


# --------------------------------------------------------------------------
# The priced, budgeted, ordered --all-pending
# --------------------------------------------------------------------------


def _seed_entries(specs: list[tuple[str, list[str]]]) -> list[tuple[str, list[str]]]:
    """One entry per spec, in order; each spec is (text, statuses of its snapshots).

    Snapshots are captured a second apart in list order; the entries are
    created in list order, which is deposit order.
    """
    import asyncio
    import uuid
    from datetime import UTC, datetime, timedelta

    from particles.core.schema import CorpusEntry, ExtractionStatus, Snapshot, WarcRecordType
    from particles.corpus.deposit import save_blob, sha256
    from particles.corpus.store import CorpusEntryRow, SnapshotRow
    from particles.db import session_scope

    base = datetime(2026, 10, 1, tzinfo=UTC)

    async def seed() -> list[tuple[str, list[str]]]:
        out: list[tuple[str, list[str]]] = []
        async with session_scope() as session:
            for i, (text, statuses) in enumerate(specs):
                entry = CorpusEntry(
                    entry_id=str(uuid.uuid4()),
                    source_type="LOCAL_MARKDOWN",
                    uri_r=f"file:///tmp/budget-{i}.md",
                    deposited_by="test",
                )
                row = CorpusEntryRow.from_model(entry)
                row.created_at = base + timedelta(minutes=i)
                session.add(row)
                snaps: list[str] = []
                for j, status in enumerate(statuses):
                    content = f"{text} (version {j})\n".encode()
                    digest = sha256(content)
                    save_blob(content, digest)
                    snap = Snapshot(
                        snapshot_id=str(uuid.uuid4()),
                        captured_at=base + timedelta(minutes=i, seconds=j),
                        content_hash=digest,
                        archive_path=str(digest),
                        extraction_status=ExtractionStatus(status),
                        warc_record_type=WarcRecordType.RESPONSE,
                    )
                    session.add(SnapshotRow.from_model(snap, entry.entry_id))
                    snaps.append(snap.snapshot_id)
                out.append((entry.entry_id, snaps))
            await session.commit()
        return out

    return asyncio.run(seed())


def _statuses() -> dict[str, str]:
    import asyncio

    from sqlalchemy import select

    from particles.corpus.store import SnapshotRow
    from particles.db import session_scope

    async def read() -> dict[str, str]:
        async with session_scope() as session:
            rows = await session.execute(
                select(SnapshotRow.snapshot_id, SnapshotRow.extraction_status)
            )
            return {sid: status for sid, status in rows.all()}

    return asyncio.run(read())


class TestAllPendingEstimate:
    def test_estimate_needs_no_key_and_makes_no_call(self, cli_db: Any, monkeypatch: Any) -> None:
        from typer.testing import CliRunner

        from particles.api.cli import app

        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        _seed_entries([("I use Notion for tasks.", ["PENDING"])])
        result = CliRunner().invoke(app, ["extract", "--all-pending", "--estimate"])

        assert result.exit_code == 0, result.output
        assert "Estimate: 1 snapshot(s) to read" in result.output
        assert "claude-sonnet-5-5 list price" in result.output
        assert "--extraction-model anthropic:claude-haiku-5-5" in result.output
        assert "--estimate: nothing was extracted." in result.output
        assert set(_statuses().values()) == {"PENDING"}

    def test_no_terminal_above_the_threshold_refuses(self, cli_db: Any, monkeypatch: Any) -> None:
        from typer.testing import CliRunner

        from particles.api.cli import app
        from particles.config import get_config

        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        _seed_entries([("I use Notion for tasks.", ["PENDING"])])
        get_config().extraction.confirm_call_threshold = 0
        result = CliRunner().invoke(app, ["extract", "--all-pending"])

        assert result.exit_code == 2
        assert "no terminal" in result.output
        assert "--budget-usd" in result.output
        assert set(_statuses().values()) == {"PENDING"}

    def test_run_flags_need_all_pending(self, cli_db: Any) -> None:
        from typer.testing import CliRunner

        from particles.api.cli import app

        for flags in (["--estimate"], ["--yes"], ["--budget-usd", "1"]):
            result = CliRunner().invoke(app, ["extract", "some-entry", *flags])
            assert result.exit_code == 1
            assert "apply to --all-pending" in result.output

    def test_an_unknown_model_route_is_refused(self, cli_db: Any) -> None:
        from typer.testing import CliRunner

        from particles.api.cli import app

        result = CliRunner().invoke(
            app, ["extract", "--all-pending", "--extraction-model", "nowhere:m"]
        )
        assert result.exit_code == 2
        assert "unknown provider" in result.output


def _billing_stub(order: list[str], output_tokens: int = 40_000) -> Any:
    """An ``extract_snapshot`` that bills ``output_tokens`` on the extraction model.

    At claude-sonnet-5-5's US$10 per MTok output, 40,000 tokens is US$0.40.
    """

    async def stub(session: Any, entry_id: str, snapshot_id: str, **_: Any) -> list[Any]:
        from particles.llm.usage import purpose_scope, record_usage

        order.append(snapshot_id)
        with purpose_scope("extraction"):
            record_usage("anthropic:claude-sonnet-5-5", output_tokens=output_tokens)
        return []

    return stub


class TestAllPendingBudget:
    def test_the_records_example_stops_after_four(self, cli_db: Any, monkeypatch: Any) -> None:
        """Estimated at US$0.50, billed US$0.40, budget US$2 (§ Acceptance)."""
        from unittest.mock import patch

        from typer.testing import CliRunner

        from particles.api.cli import app
        from particles.operations import extract_run
        from particles.operations.spend_estimate import ExtractionRange

        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        seeded = _seed_entries([(f"Note {i}.", ["PENDING"]) for i in range(10)])
        real_plan_run = extract_run.plan_run

        async def priced(session: Any, pending: Any) -> extract_run.RunPlan:
            plan = await real_plan_run(session, pending)
            plan.high_usd = {item.snapshot_id: 0.50 for item in pending}
            plan.total = ExtractionRange(10, "claude-sonnet-5-5", 4.0, 5.0, 9.0, 60.0)
            return plan

        order: list[str] = []
        with (
            patch("particles.operations.extract_run.plan_run", priced),
            patch("particles.operations.extract.extract_snapshot", _billing_stub(order)),
        ):
            result = CliRunner().invoke(app, ["extract", "--all-pending", "--budget-usd", "2"])

        assert result.exit_code == 0, result.output
        assert order == [snaps[0] for _, snaps in seeded[:4]]
        assert "Budget reached: spent US$1.60 of US$2.00. 6 snapshot(s)" in result.output
        assert "Budget: US$2.00" in result.output

    def test_a_budget_is_consent_without_a_terminal(self, cli_db: Any, monkeypatch: Any) -> None:
        from unittest.mock import patch

        from typer.testing import CliRunner

        from particles.api.cli import app
        from particles.config import get_config

        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        _seed_entries([("Note.", ["PENDING"])])
        get_config().extraction.confirm_call_threshold = 0
        order: list[str] = []
        with patch(
            "particles.operations.extract.extract_snapshot", _billing_stub(order, output_tokens=10)
        ):
            result = CliRunner().invoke(app, ["extract", "--all-pending", "--budget-usd", "5"])
        assert result.exit_code == 0, result.output
        assert len(order) == 1


class TestAllPendingOrder:
    def test_an_edit_to_a_read_note_goes_before_the_unread_backlog(
        self, cli_db: Any, monkeypatch: Any
    ) -> None:
        """Deposit order reads the edit last; edit-first order reads it first."""
        from unittest.mock import patch

        from typer.testing import CliRunner

        from particles.api.cli import app

        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        seeded = _seed_entries(
            [
                ("Unread one.", ["PENDING"]),
                ("Unread two.", ["PENDING"]),
                ("Read, then edited.", ["COMPLETE", "PENDING"]),
            ]
        )
        order: list[str] = []
        with patch(
            "particles.operations.extract.extract_snapshot", _billing_stub(order, output_tokens=10)
        ):
            result = CliRunner().invoke(app, ["extract", "--all-pending"])

        assert result.exit_code == 0, result.output
        edited = seeded[2][1][1]
        assert order == [edited, seeded[0][1][0], seeded[1][1][0]]

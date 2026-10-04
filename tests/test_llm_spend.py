# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""``particles extract`` reports and records what it spent.

Extraction is the bulk of a document store's LLM bill, and it opened no usage
scope, so the verb that spends most reported nothing. Each route now meters the
run, prints the ``LLM usage:`` line on stderr, and appends one ``EXTRACT_RUN``
event; ``quality`` and the digest sum those events with the consolidation runs
into one recorded-spend line.

The Anthropic client is mocked through the ``set_client`` seam with replies
carrying known ``usage`` values, so every token count and dollar figure below
is exact.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import anthropic
import pytest
from typer.testing import CliRunner

from particles.config import ProviderSelection, get_config
from particles.core.schema import ExtractionStatus, StoreLLMSpend, WarcRecordType
from particles.llm.usage import LLMUsage, UsageRow, render_store_spend_line
from particles.store.event_store import OperatorEvent, OperatorEventType
from tests._client_fixtures import stream_via_create

#: claude-haiku-4-5 lists at $1 / $5 per MTok, so a 100k-in / 10k-out reply
#: costs exactly $0.15.
_MODEL = "claude-haiku-4-5"
_INPUT = 100_000
_OUTPUT = 10_000
_COST_PER_CALL = 0.15


def _reply() -> SimpleNamespace:
    text = json.dumps(
        [
            {
                "content": "The speed of light in vacuum is about 299,792 km/s.",
                "confidence_value": 0.97,
                "uncertainty_nature": "EPISTEMIC",
            }
        ]
    )
    return SimpleNamespace(
        content=[SimpleNamespace(text=text)],
        stop_reason="end_turn",
        usage=SimpleNamespace(
            input_tokens=_INPUT,
            output_tokens=_OUTPUT,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
        ),
    )


@pytest.fixture
def paying_client(cli_db: Path, no_embedding_model: None) -> Any:
    """A mocked Anthropic client whose every reply reports the same usage."""
    from particles.llm import set_client

    get_config().llm.extraction = ProviderSelection(provider="anthropic", model=_MODEL)
    client = MagicMock(spec=anthropic.Anthropic)
    client.messages = MagicMock()
    client.messages.create = MagicMock(side_effect=lambda **_k: _reply())
    stream_via_create(client)
    set_client(client)
    yield client
    set_client(None)


async def _seed_pending(count: int) -> list[tuple[str, str]]:
    """``count`` PENDING snapshots, one per corpus entry; ``(entry_id, snapshot_id)``."""
    from particles.core.schema import CorpusEntry, Snapshot
    from particles.corpus.deposit import save_blob, sha256
    from particles.corpus.store import CorpusEntryRow, SnapshotRow
    from particles.db import session_scope

    ids: list[tuple[str, str]] = []
    async with session_scope() as session:
        for i in range(count):
            content = f"Fact number {i}: the speed of light is about 299,792 km/s.\n".encode()
            digest = sha256(content)
            save_blob(content, digest)
            entry = CorpusEntry(
                entry_id=str(uuid.uuid4()),
                source_type="LOCAL_MARKDOWN",
                uri_r=f"file:///tmp/fact-{i}.md",
                deposited_by="test",
            )
            snap = Snapshot(
                snapshot_id=str(uuid.uuid4()),
                captured_at=datetime.now(UTC),
                content_hash=digest,
                archive_path=str(digest),
                extraction_status=ExtractionStatus.PENDING,
                warc_record_type=WarcRecordType.RESPONSE,
            )
            session.add(CorpusEntryRow.from_model(entry))
            session.add(SnapshotRow.from_model(snap, entry.entry_id))
            ids.append((entry.entry_id, snap.snapshot_id))
        await session.commit()
    return ids


async def _extract_runs() -> list[OperatorEvent]:
    from particles.db import session_scope
    from particles.store.event_store import list_events

    async with session_scope() as session:
        return await list_events(session, event_type=OperatorEventType.EXTRACT_RUN)


def _calls_in(event: OperatorEvent) -> int:
    return sum(row["calls"] for row in (event.payload or {})["llm_usage"]["rows"])


# ---------------------------------------------------------------------------
# Every route prints the line and records the run
# ---------------------------------------------------------------------------


class TestExtractRoutesReportUsage:
    def test_single_snapshot_prints_the_line_on_stderr_and_records_the_run(
        self, paying_client: Any
    ) -> None:
        from particles.api.cli import app

        [(entry_id, snapshot_id)] = asyncio.run(_seed_pending(1))
        result = CliRunner().invoke(app, ["extract", entry_id, "--snapshot-id", snapshot_id])

        assert result.exit_code == 0, result.output
        calls = paying_client.messages.create.call_count
        assert calls >= 1
        # stderr, so stdout stays the extraction report (the audit precedent).
        assert "LLM usage:" in result.stderr
        assert "LLM usage:" not in result.stdout
        assert f"({_MODEL})" in result.stderr
        assert "at list price" in result.stderr

        [event] = asyncio.run(_extract_runs())
        assert event.actor == "cli:extract"
        payload = event.payload or {}
        assert (payload["route"], payload["snapshots"]) == ("single", 1)
        assert _calls_in(event) == calls
        assert payload["llm_usage"]["cost_usd"] == pytest.approx(calls * _COST_PER_CALL)

    def test_all_pending_prints_one_line_and_records_one_run(self, paying_client: Any) -> None:
        from particles.api.cli import app

        asyncio.run(_seed_pending(3))
        result = CliRunner().invoke(app, ["extract", "--all-pending"])

        assert result.exit_code == 0, result.output
        calls = paying_client.messages.create.call_count
        assert calls >= 3
        # One line for the whole pass, not one per snapshot.
        assert result.stderr.count("LLM usage:") == 1
        [event] = asyncio.run(_extract_runs())
        payload = event.payload or {}
        assert (payload["route"], payload["snapshots"]) == ("all-pending", 3)
        assert _calls_in(event) == calls
        assert payload["llm_usage"]["cost_usd"] == pytest.approx(calls * _COST_PER_CALL)

    def test_all_pending_stopped_early_still_prints_the_line(self, cli_db: Path) -> None:
        """An account-level stop exits 1, and the line is printed on the way out."""
        from particles.api.cli import app
        from particles.llm import set_client

        class _CreditError(Exception):
            status_code = 400
            message = "Your credit balance is too low to access the Anthropic API."

        asyncio.run(_seed_pending(2))
        client = MagicMock()
        client.messages = MagicMock()
        client.messages.create = MagicMock(side_effect=_CreditError(_CreditError.message))
        stream_via_create(client)
        set_client(client)
        try:
            result = CliRunner().invoke(app, ["extract", "--all-pending"])
        finally:
            set_client(None)

        assert result.exit_code == 1
        # A refused call is not billed: the line says so, and nothing is recorded.
        assert "LLM usage: no LLM calls." in result.stderr
        assert asyncio.run(_extract_runs()) == []

    def test_nothing_pending_records_nothing(
        self, cli_db: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A hook that runs ``extract --all-pending`` every turn must not fill the log."""
        from particles.api.cli import app

        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        result = CliRunner().invoke(app, ["extract", "--all-pending"])
        assert result.exit_code == 0
        assert "LLM usage:" not in result.output
        assert asyncio.run(_extract_runs()) == []

    def test_post_extract_records_the_run(self, paying_client: Any) -> None:
        """A remote caller's extraction counts toward the engine store's spend."""
        from fastapi.testclient import TestClient

        from particles.api.app import app

        [(entry_id, snapshot_id)] = asyncio.run(_seed_pending(1))
        client = TestClient(app, client=("127.0.0.1", 50000))
        resp = client.post("/extract", json={"entry_id": entry_id, "snapshot_id": snapshot_id})

        assert resp.status_code == 200, resp.text
        [event] = asyncio.run(_extract_runs())
        assert event.actor == "http:/extract"
        assert (event.payload or {})["route"] == "http"
        assert _calls_in(event) == paying_client.messages.create.call_count

    def test_remote_backend_prints_no_line(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Over HTTP the engine made and recorded the calls; the laptop has none."""
        from particles.api.cli.extract import _echo_usage

        _echo_usage(None)
        assert capsys.readouterr().err == ""


# ---------------------------------------------------------------------------
# The per-store cumulative figure
# ---------------------------------------------------------------------------


def _usage(cost: float | None, calls: int = 1) -> dict[str, Any]:
    rows = [UsageRow(purpose="extraction", provider="anthropic", model=_MODEL, calls=calls)]
    return LLMUsage(
        rows=rows if calls else [],
        cost_usd=cost,
        unpriced=[] if cost is not None else [f"anthropic:{_MODEL}"],
    ).model_dump(mode="json")


async def _record(
    event_type: OperatorEventType, payload: dict[str, Any], at: datetime | None = None
) -> None:
    from particles.db import session_scope
    from particles.store.event_store import OperatorEventRow, record_event

    async with session_scope() as session:
        event = await record_event(session, actor="test", event_type=event_type, payload=payload)
        if at is not None:
            row = await session.get(OperatorEventRow, event.event_id)
            assert row is not None
            row.occurred_at = at
        await session.commit()


async def _spend() -> StoreLLMSpend | None:
    from particles.db import session_scope
    from particles.operations.llm_spend import store_llm_spend

    async with session_scope() as session:
        return await store_llm_spend(session)


class TestStoreSpend:
    def test_none_before_any_run_recorded_usage(self, cli_db: Path) -> None:
        assert asyncio.run(_spend()) is None

    def test_is_the_sum_of_the_recorded_runs(self, cli_db: Path) -> None:
        first = datetime(2026, 9, 1, 12, tzinfo=UTC)
        asyncio.run(
            _record(OperatorEventType.CONSOLIDATION_RUN, {"llm_usage": _usage(1.25)}, first)
        )
        asyncio.run(
            _record(
                OperatorEventType.EXTRACT_RUN,
                {"llm_usage": _usage(2.50)},
                first + timedelta(days=3),
            )
        )
        asyncio.run(_record(OperatorEventType.EXTRACT_RUN, {"llm_usage": _usage(0.75)}))

        spend = asyncio.run(_spend())
        assert spend is not None
        assert spend.cost_usd == pytest.approx(4.50)
        assert spend.runs == 3
        assert spend.since.date().isoformat() == "2026-09-01"
        assert spend.unpriced_runs == 0

    def test_runs_without_calls_or_usage_are_not_runs(self, cli_db: Path) -> None:
        """A structural-only cycle, or one recorded before usage existed, spent nothing here."""
        asyncio.run(_record(OperatorEventType.CONSOLIDATION_RUN, {"census": {}}))
        asyncio.run(_record(OperatorEventType.CONSOLIDATION_RUN, {"llm_usage": _usage(0.0, 0)}))
        asyncio.run(_record(OperatorEventType.EXTRACT_RUN, {"llm_usage": _usage(0.40)}))
        # Another event type carrying the key is not a run record.
        asyncio.run(_record(OperatorEventType.TRUST_CHANGED, {"llm_usage": _usage(9.0)}))

        spend = asyncio.run(_spend())
        assert spend is not None
        assert (spend.runs, spend.cost_usd) == (1, pytest.approx(0.40))

    def test_an_unpriced_run_is_counted_and_disclosed_never_summed_at_zero(
        self, cli_db: Path
    ) -> None:
        asyncio.run(_record(OperatorEventType.EXTRACT_RUN, {"llm_usage": _usage(1.00)}))
        asyncio.run(_record(OperatorEventType.EXTRACT_RUN, {"llm_usage": _usage(None)}))

        spend = asyncio.run(_spend())
        assert spend is not None
        assert (spend.runs, spend.unpriced_runs) == (2, 1)
        assert spend.cost_usd == pytest.approx(1.00)
        line = render_store_spend_line(spend)
        assert "1 run used a model with no llm.price_per_mtok entry" in line

    def test_extract_runs_join_the_sum(self, paying_client: Any) -> None:
        """End to end: a real (mocked) extraction lands in the store figure."""
        from particles.api.cli import app

        asyncio.run(_record(OperatorEventType.CONSOLIDATION_RUN, {"llm_usage": _usage(1.00)}))
        asyncio.run(_seed_pending(2))
        result = CliRunner().invoke(app, ["extract", "--all-pending"])
        assert result.exit_code == 0, result.output

        calls = paying_client.messages.create.call_count
        spend = asyncio.run(_spend())
        assert spend is not None
        assert spend.runs == 2
        assert spend.cost_usd == pytest.approx(1.00 + calls * _COST_PER_CALL)


class TestSpendLine:
    def test_wording(self) -> None:
        spend = StoreLLMSpend(
            cost_usd=41.2, runs=37, since=datetime(2026, 9, 30, tzinfo=UTC), unpriced_runs=0
        )
        assert render_store_spend_line(spend) == (
            "LLM spend recorded for this store: US$41 across 37 runs since 2026-09-30."
        )

    def test_one_run(self) -> None:
        spend = StoreLLMSpend(cost_usd=0.15, runs=1, since=datetime(2026, 9, 30, tzinfo=UTC))
        assert render_store_spend_line(spend) == (
            "LLM spend recorded for this store: US$0.15 across 1 run since 2026-09-30."
        )

    def test_quality_shows_the_line(self, cli_db: Path) -> None:
        from particles.api.cli import app

        asyncio.run(_record(OperatorEventType.EXTRACT_RUN, {"llm_usage": _usage(2.00)}))
        result = CliRunner().invoke(app, ["quality"])
        assert result.exit_code == 0, result.output
        assert "LLM spend recorded for this store: US$2.00 across 1 run since " in result.output

    def test_quality_omits_the_line_before_any_run(self, cli_db: Path) -> None:
        from particles.api.cli import app

        result = CliRunner().invoke(app, ["quality"])
        assert result.exit_code == 0, result.output
        assert "LLM spend" not in result.output

    def test_digest_shows_the_line(self, cli_db: Path, no_embedding_model: None) -> None:
        from particles.db import DEFAULT_STORE
        from particles.operations.digest import build_digest

        before = asyncio.run(build_digest(DEFAULT_STORE))
        assert "LLM spend" not in before

        asyncio.run(_record(OperatorEventType.EXTRACT_RUN, {"llm_usage": _usage(2.00)}))
        after = asyncio.run(build_digest(DEFAULT_STORE))
        assert after.rstrip().endswith(
            f"_LLM spend recorded for this store: US$2.00 across 1 run since "
            f"{datetime.now(UTC).date().isoformat()}._"
        )
        # Additive: the digest above the line is unchanged.
        assert after.startswith(before.rstrip())


# ---------------------------------------------------------------------------
# reindex and structure record through the same meter
# ---------------------------------------------------------------------------

#: What a reindex run reports when it visited three snapshots.
_REINDEX_SUMMARY: dict[str, Any] = {
    "scope": 3,
    "succeeded": 2,
    "failed": 1,
    "failed_entries": ["e3"],
    "lint_summary": {},
    "dry_run": False,
    "plan": None,
}


def _paying_reindex(summary: dict[str, Any]) -> Any:
    """A stand-in for the reindex operation that pays for one extraction call.

    Carry-forward would let a real re-extraction of unchanged text make no call
    at all; the wiring under test is the meter around the operation, so the
    stand-in makes one real call through the mocked port and returns ``summary``.
    """

    async def _reindex(*_args: Any, **kwargs: Any) -> dict[str, Any]:
        from particles.llm import complete

        if not kwargs.get("dry_run"):
            await complete("extraction", "re-extract this", max_tokens=100)
        return dict(summary)

    return _reindex


class TestReindexAndStructureRecordSpend:
    def test_cli_reindex_prints_the_line_and_records_a_reindex_run(
        self, paying_client: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from particles.api.cli import app

        monkeypatch.setattr(
            "particles.operations.reindex.reindex", _paying_reindex(_REINDEX_SUMMARY)
        )
        result = CliRunner().invoke(app, ["reindex", "--format", "json"])

        assert result.exit_code == 0, result.output
        assert result.stderr.count("LLM usage:") == 1
        assert f"({_MODEL})" in result.stderr
        assert "LLM usage:" not in result.stdout
        [event] = asyncio.run(_extract_runs())
        assert event.actor == "cli:reindex"
        payload = event.payload or {}
        # One EXTRACT_RUN under its own route: no new event type to sum.
        assert (payload["route"], payload["snapshots"]) == ("reindex", 3)
        assert _calls_in(event) == 1
        assert payload["llm_usage"]["cost_usd"] == pytest.approx(_COST_PER_CALL)

        spend = asyncio.run(_spend())
        assert spend is not None
        assert (spend.runs, spend.cost_usd) == (1, pytest.approx(_COST_PER_CALL))

    def test_reindex_dry_run_prints_no_line_and_records_nothing(
        self, paying_client: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from particles.api.cli import app

        monkeypatch.setattr(
            "particles.operations.reindex.reindex",
            _paying_reindex({**_REINDEX_SUMMARY, "dry_run": True}),
        )
        result = CliRunner().invoke(app, ["reindex", "--dry-run", "--format", "json"])

        assert result.exit_code == 0, result.output
        assert "LLM usage:" not in result.output
        assert paying_client.messages.create.call_count == 0
        assert asyncio.run(_extract_runs()) == []

    def test_post_reindex_records_the_run(
        self, paying_client: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from fastapi.testclient import TestClient

        from particles.api.app import app

        monkeypatch.setattr("particles.api.app.reindex", _paying_reindex(_REINDEX_SUMMARY))
        client = TestClient(app, client=("127.0.0.1", 50000))
        resp = client.post("/reindex", json={})

        assert resp.status_code == 200, resp.text
        [event] = asyncio.run(_extract_runs())
        assert event.actor == "http:/reindex"
        assert ((event.payload or {})["route"], (event.payload or {})["snapshots"]) == (
            "reindex",
            3,
        )

    def test_post_reindex_dry_run_records_nothing(
        self, paying_client: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from fastapi.testclient import TestClient

        from particles.api.app import app

        monkeypatch.setattr(
            "particles.api.app.reindex", _paying_reindex({**_REINDEX_SUMMARY, "dry_run": True})
        )
        client = TestClient(app, client=("127.0.0.1", 50000))
        resp = client.post("/reindex", json={"dry_run": True})

        assert resp.status_code == 200, resp.text
        assert asyncio.run(_extract_runs()) == []

    def test_structure_records_a_structure_run(
        self, paying_client: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from particles.api.cli import app

        async def _backfill(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
            from particles.llm import complete

            await complete("extraction", "structure this", max_tokens=100)
            return {"scope": 1, "annotated": 1, "skipped": 0, "failed": 0}

        monkeypatch.setattr("particles.operations.structure.backfill_structured_claims", _backfill)
        result = CliRunner().invoke(app, ["structure"])

        assert result.exit_code == 0, result.output
        assert "LLM usage:" in result.stderr
        [event] = asyncio.run(_extract_runs())
        assert event.actor == "cli:structure"
        assert (event.payload or {})["route"] == "structure"
        assert _calls_in(event) == 1

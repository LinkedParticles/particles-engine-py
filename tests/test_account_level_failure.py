"""Account-level LLM failures abort a bulk run instead of repeating.

The dogfood incident this fixes (2026-07-25): with an exhausted credit balance,
``particles extract --all-pending`` walked all 68 pending snapshots — and every
page of each PDF — re-issuing a request that could not succeed, printing the
same billing 400 hundreds of lines deep. The key was *present*, so the existing
fail-fast guard (which only covers a missing key) did not fire, and the
extraction seam classified the billing error as a per-call transient.

The predicate that distinguishes the two already existed for the circuit breaker, but it lived in the Engine (``operations/_llm``) where the
Client-layer extraction seam could not reach it.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from typer.testing import CliRunner

from particles.core.schema import ExtractionStatus, WarcRecordType
from particles.llm.errors import AccountLevelLLMError, is_account_level_failure


class _ApiError(Exception):
    """Stand-in for an SDK error carrying ``status_code`` (duck-typed)."""

    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code


_CREDIT = "Your credit balance is too low to access the Anthropic API."


class TestPredicate:
    """The account-level / per-call split. Wrong either way is a real cost:
    a false positive aborts a healthy run; a false negative is this incident."""

    @pytest.mark.parametrize(
        "exc",
        [
            _ApiError(401, "invalid x-api-key"),
            _ApiError(403, "permission denied"),
            _ApiError(400, _CREDIT),
            _ApiError(400, "Billing issue: add a payment method"),
        ],
    )
    def test_account_level(self, exc: Exception) -> None:
        assert is_account_level_failure(exc) is True

    @pytest.mark.parametrize(
        "exc",
        [
            _ApiError(400, "messages.0: content blocks must not be empty"),
            _ApiError(429, "rate limit exceeded"),
            _ApiError(500, "internal server error"),
            _ApiError(529, "overloaded"),
            RuntimeError("connection reset by peer"),
        ],
    )
    def test_per_call_or_transient(self, exc: Exception) -> None:
        assert is_account_level_failure(exc) is False

    def test_engine_breaker_shares_the_predicate(self) -> None:
        """One definition — the Engine breaker and the Client seam cannot drift."""
        from particles.operations import _llm

        assert _llm._is_account_level is is_account_level_failure


class TestExtractionSeamRaises:
    @pytest.mark.asyncio
    async def test_credit_error_raises_rather_than_degrading(self) -> None:
        """The seam used to return ``transient=True``, which is what made the
        caller retry the next snapshot, and the next, and the next."""
        from particles.extraction import general

        async def boom(*_a: Any, **_k: Any) -> str:
            raise _ApiError(400, _CREDIT)

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("particles.llm.complete_with_provider_model", boom)
            with pytest.raises(AccountLevelLLMError):
                await general._call_llm("some source text")

    @pytest.mark.asyncio
    async def test_per_call_error_still_degrades_to_transient(self) -> None:
        """Unchanged for the failure class that genuinely is worth retrying."""
        from particles.extraction import general

        async def boom(*_a: Any, **_k: Any) -> str:
            raise _ApiError(429, "rate limit exceeded")

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("particles.llm.complete_with_provider_model", boom)
            candidates, notes, transient = await general._call_llm("some source text")

        assert candidates == []
        assert transient is True
        assert any("API error" in n for n in notes)

    @pytest.mark.asyncio
    async def test_journal_seam_matches(self) -> None:
        from particles.extraction import journal

        async def boom(*_a: Any, **_k: Any) -> str:
            raise _ApiError(400, _CREDIT)

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("particles.llm.complete_with_provider_model", boom)
            with pytest.raises(AccountLevelLLMError):
                await journal._call_journal_llm("a journal entry")


# --------------------------------------------------------------------------
# The CLI reproduction
# --------------------------------------------------------------------------


async def _seed_pending(count: int) -> list[str]:
    """``count`` PENDING snapshots across ``count`` corpus entries."""
    from particles.core.schema import CorpusEntry, Snapshot
    from particles.corpus.deposit import save_blob, sha256
    from particles.corpus.store import CorpusEntryRow, SnapshotRow
    from particles.db import session_scope

    ids: list[str] = []
    async with session_scope() as session:
        for i in range(count):
            content = f"Rule number {i}: something durable.\n".encode()
            digest = sha256(content)
            save_blob(content, digest)
            entry = CorpusEntry(
                entry_id=str(uuid.uuid4()),
                source_type="LOCAL_MARKDOWN",
                uri_r=f"file:///tmp/rule-{i}.md",
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
            ids.append(snap.snapshot_id)
        await session.commit()
    return ids


async def _statuses(snapshot_ids: list[str]) -> list[str]:
    from particles.corpus.store import SnapshotRow
    from particles.db import session_scope

    async with session_scope() as session:
        return [
            (await session.get(SnapshotRow, sid)).extraction_status  # type: ignore[union-attr]
            for sid in snapshot_ids
        ]


class TestExtractAllPendingAborts:
    def test_stops_on_the_first_snapshot_and_leaves_the_queue_pending(
        self, cli_db: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The incident, reproduced: five pending snapshots, an exhausted key.

        Before the fix this printed the same billing error five times (and would
        have gone on for all 68 in the real run). Now: one message, one
        remaining-count, and every snapshot still PENDING for the retry.
        """
        from particles.api.cli import app
        from particles.llm import set_client

        snapshot_ids = asyncio.run(_seed_pending(5))

        calls = {"n": 0}

        def create(*_a: Any, **_k: Any) -> Any:
            calls["n"] += 1
            raise _ApiError(400, _CREDIT)

        client = MagicMock()
        client.messages = MagicMock()
        client.messages.create = MagicMock(side_effect=create)
        set_client(client)
        try:
            result = CliRunner().invoke(app, ["extract", "--all-pending"])
        finally:
            set_client(None)

        assert result.exit_code == 1
        assert "LLM unavailable (account-level)" in result.output
        assert "credit balance" in result.output
        # Nothing completed, so the honest count is 0 done / all 5 still queued.
        assert "Stopped after 0 of 5" in result.output
        assert "5 still PENDING" in result.output
        # Exactly one API call — not one per snapshot.
        assert calls["n"] == 1
        # Nothing was lost or stranded IN_PROGRESS.
        assert asyncio.run(_statuses(snapshot_ids)) == [ExtractionStatus.PENDING.value] * 5

    def test_the_message_is_not_repeated_per_snapshot(
        self, cli_db: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The reported symptom was volume: hundreds of lines for one condition."""
        from particles.api.cli import app
        from particles.llm import set_client

        asyncio.run(_seed_pending(8))

        client = MagicMock()
        client.messages = MagicMock()
        client.messages.create = MagicMock(side_effect=_ApiError(400, _CREDIT))
        set_client(client)
        try:
            result = CliRunner().invoke(app, ["extract", "--all-pending"])
        finally:
            set_client(None)

        # The provider's message appears exactly once, however many snapshots
        # were queued. (The remediation line mentions "credit balance" too, so
        # count the provider text specifically.)
        assert result.output.count(_CREDIT) == 1


class TestConsolidationPassStops:
    @pytest.mark.asyncio
    async def test_extract_pass_breaks_instead_of_burning_the_cap(
        self, db_session: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """§8 is continue-and-disclose per *snapshot*; this is not per-snapshot.

        Pins the serial loop (``consolidation.extract_batching: false``); the
        pooled default's equivalent — every task fails on the ONE merged
        dispatch, disclosed once — is the test below.
        """
        from particles.config import get_config
        from particles.operations import consolidation

        get_config().consolidation.extract_batching = False
        seen = {"n": 0}

        async def boom(*_a: Any, **_k: Any) -> list[Any]:
            seen["n"] += 1
            raise AccountLevelLLMError(_ApiError(400, _CREDIT))

        monkeypatch.setattr("particles.operations.extract.extract_snapshot", boom)
        monkeypatch.setattr(
            "particles.corpus.store.list_pending_snapshots_oldest_first",
            _fake_pending(6),
        )

        report = consolidation.ConsolidationReport(actor="test")
        extracted = await consolidation._pass_extract(db_session, report)

        assert extracted == 0
        assert seen["n"] == 1  # stopped after the first, did not burn the cap
        assert report.pending_failed == 1

    @pytest.mark.asyncio
    async def test_pooled_extract_pass_discloses_the_account_failure_once(
        self, db_session: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """under pooling every task shares ONE merged dispatch, so
        an account-level failure reaches each parked task simultaneously — the
        pass discloses it once, exactly as the serial break did. (That only one
        API call is issued is the pool's contract, pinned in
        tests/test_llm_pool.py and tests/test_llm_batch.py.)"""
        from contextlib import asynccontextmanager

        import particles.db as db_mod
        from particles.operations import consolidation

        @asynccontextmanager
        async def fake_scope(*_a: Any, **_k: Any) -> Any:
            yield AsyncMock()

        monkeypatch.setattr(db_mod, "session_scope", fake_scope)

        async def boom(*_a: Any, **_k: Any) -> list[Any]:
            raise AccountLevelLLMError(_ApiError(400, _CREDIT))

        monkeypatch.setattr("particles.operations.extract.extract_snapshot", boom)
        monkeypatch.setattr(
            "particles.corpus.store.list_pending_snapshots_oldest_first",
            _fake_pending(6),
        )

        report = consolidation.ConsolidationReport(actor="test")
        extracted = await consolidation._pass_extract(db_session, report)

        assert extracted == 0
        assert report.pending_extracted == 0
        assert report.pending_failed == 1  # one disclosure, not six


def _fake_pending(count: int) -> Any:
    async def _listed(_session: Any) -> list[tuple[str, str]]:
        return [(f"entry-{i}", f"snap-{i}") for i in range(count)]

    return _listed

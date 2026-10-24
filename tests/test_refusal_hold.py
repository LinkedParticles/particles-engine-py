# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""A source the extraction model refuses is recorded, held and shown.

Every model call is a scripted in-process provider routed with
``override_providers`` (or a mocked SDK client for the adapter cases), so
nothing here reaches a live LLM.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from particles.core.schema import ExtractionStatus, Mutability
from particles.corpus.deposit import deposit_file
from particles.corpus.refusals import ExtractionRefusalRow, list_held_snapshots
from particles.corpus.store import SnapshotRow
from particles.ingest.pipeline import SnapshotOutcome, extract_snapshot
from particles.ingest.refusal_hold import current_refusal_pairings, held_snapshots
from particles.llm import CompletionRefused, RequestFailure, override_providers
from tests._client_fixtures import stream_via_create

_REPLY = json.dumps(
    [
        {
            "content": "LUKS encrypts the disk.",
            "confidence_value": 0.9,
            "uncertainty_nature": "EPISTEMIC",
        }
    ]
)


class _Scripted:
    """A provider that plays a script: ``"refuse"``, ``"fail"`` or a reply text."""

    def __init__(self, script: Sequence[str], provider_model: str | None = None) -> None:
        self._script = list(script)
        self._model = provider_model or current_refusal_pairings()["extraction"]
        self.calls = 0

    @property
    def provider_model(self) -> str:
        return self._model

    async def complete(self, prompt: str, **_: object) -> str:
        step = self._script[min(self.calls, len(self._script) - 1)]
        self.calls += 1
        if step == "refuse":
            raise CompletionRefused(
                "declined (stop_reason=refusal, category=cyber)",
                provider_model=self._model,
                category="cyber",
            )
        if step == "fail":
            raise RuntimeError("Error code: 529 - overloaded")
        return step


@pytest.fixture
def mock_encoder() -> Iterator[None]:
    from particles import embeddings as ep

    model = MagicMock()
    model.encode = MagicMock(return_value=[[0.1, 0.2, 0.3, 0.4]])
    original = ep._embedding_model
    ep.set_embedding_model(model)
    try:
        yield
    finally:
        ep.set_embedding_model(original)


async def _deposit(
    session: AsyncSession, tmp_path: Path, text: str = "LUKS notes.", **kw: Any
) -> tuple[str, str]:
    path = tmp_path / kw.pop("name", "luks.md")
    path.write_text(text)
    entry_id, snapshot_id = await deposit_file(session, path, deposited_by="test", **kw)
    await session.commit()
    return entry_id, snapshot_id


async def _extract(
    session: AsyncSession, entry_id: str, snapshot_id: str, provider: _Scripted
) -> SnapshotOutcome:
    outcome = SnapshotOutcome()
    with override_providers({"extraction": provider}):
        await extract_snapshot(session, entry_id, snapshot_id, outcome_out=outcome)
    await session.commit()
    return outcome


async def _rows(session: AsyncSession) -> list[ExtractionRefusalRow]:
    return list((await session.execute(select(ExtractionRefusalRow))).scalars())


def _big_text() -> str:
    block = "This is a sentence about disk encryption. " * 500
    return "\n\n".join(block for _ in range(4))


# ---------------------------------------------------------------------------
# §1: the typed refusal
# ---------------------------------------------------------------------------


class TestTypedRefusal:
    @pytest.mark.asyncio
    async def test_anthropic_refusal_raises_completion_refused(self) -> None:
        import anthropic

        from particles.llm import complete, set_client

        resp = MagicMock()
        resp.stop_reason = "refusal"
        resp.stop_details = SimpleNamespace(category="cyber")
        resp.content = []
        resp.usage = SimpleNamespace(
            input_tokens=5,
            output_tokens=0,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=0,
        )
        client = MagicMock(spec=anthropic.Anthropic)
        client.messages = MagicMock()
        client.messages.create = MagicMock(return_value=resp)
        stream_via_create(client)
        set_client(client)
        try:
            with pytest.raises(CompletionRefused) as info:
                await complete("extraction", "LUKS", max_tokens=100)
        finally:
            set_client(None)
        assert info.value.category == "cyber"
        assert info.value.provider_model.startswith("anthropic:")

    def test_openai_compat_content_filter_is_a_refusal(self) -> None:
        from particles.llm.adapters.openai_compat import _extract_text

        payload = {"choices": [{"finish_reason": "content_filter", "message": {"content": ""}}]}
        with pytest.raises(CompletionRefused) as info:
            _extract_text(payload, provider_model="local:qwen3")
        assert info.value.provider_model == "local:qwen3"

    @pytest.mark.asyncio
    async def test_sequential_fallback_reports_refused(self) -> None:
        from particles.llm.registry import CompletionRequest, run_sequential

        texts, failures = await run_sequential(
            _Scripted(["refuse"]),  # type: ignore[arg-type]
            [CompletionRequest(prompt="x")],
            max_tokens=10,
            temperature=None,
            response_schema=None,
        )
        assert texts == [None]
        assert failures == [RequestFailure.REFUSED]

    @pytest.mark.asyncio
    async def test_a_refusal_is_not_retried_at_the_larger_budget(self) -> None:
        from particles.extraction.general import _call_llm, tally_replies

        provider = _Scripted(["refuse"])
        with override_providers({"extraction": provider}), tally_replies() as tally:  # type: ignore[dict-item]
            candidates, _, transient = await _call_llm("LUKS")
        assert candidates == [] and transient
        assert provider.calls == 1
        assert tally.refused == 1 and tally.answered == 0
        assert tally.refusals[0].category == "cyber"

    @pytest.mark.asyncio
    async def test_pooled_refusal_is_recorded_and_not_retried(self, monkeypatch: Any) -> None:
        from particles.extraction import general

        async def fake_group(
            pool: object,
            planned: object,
            *,
            max_tokens: int | None = None,
            failures_out: Any = None,
        ) -> tuple[list[str | None], str]:
            if failures_out is not None:
                failures_out.append(RequestFailure.REFUSED)
            if max_tokens is not None:
                raise AssertionError("a refused request must not ride the retry wave")
            return [None], "anthropic:m"

        monkeypatch.setattr(general, "_pooled_group_complete", fake_group)
        planned = general._build_llm_request("LUKS")
        with general.tally_replies() as tally:
            [(candidates, notes, transient)] = await general._pooled_extract(object(), [planned])  # type: ignore[arg-type]
        assert candidates == [] and transient
        assert tally.refused == 1


# ---------------------------------------------------------------------------
# §3: what a pass records
# ---------------------------------------------------------------------------


class TestRecord:
    @pytest.mark.asyncio
    async def test_case_1_single_pass_refusal_is_recorded(
        self, db_session: AsyncSession, tmp_path: Path
    ) -> None:
        entry_id, snapshot_id = await _deposit(db_session, tmp_path)
        outcome = await _extract(db_session, entry_id, snapshot_id, _Scripted(["refuse"]))

        snap = await db_session.get(SnapshotRow, snapshot_id)
        assert snap is not None and snap.extraction_status == ExtractionStatus.PENDING.value
        assert outcome.refused_calls == 1
        assert outcome.refusal_category == "cyber"
        assert outcome.refusal_recorded
        [row] = await _rows(db_session)
        assert row.content_hash == snap.content_hash
        assert row.purpose == "extraction"
        assert row.provider_model == current_refusal_pairings()["extraction"]
        assert row.snapshot_id == snapshot_id and row.category == "cyber"

    @pytest.mark.asyncio
    async def test_case_9_refusal_beside_a_transient_failure_records_nothing(
        self, db_session: AsyncSession, tmp_path: Path, mock_encoder: None
    ) -> None:
        entry_id, snapshot_id = await _deposit(db_session, tmp_path, _big_text(), name="big.md")
        outcome = await _extract(db_session, entry_id, snapshot_id, _Scripted(["refuse", "fail"]))
        assert outcome.refused_calls == 1 and outcome.failed_calls > 1
        assert not outcome.refusal_recorded
        assert await _rows(db_session) == []

    @pytest.mark.asyncio
    async def test_case_10_partial_refusal_is_reported_not_held(
        self, db_session: AsyncSession, tmp_path: Path, mock_encoder: None
    ) -> None:
        entry_id, snapshot_id = await _deposit(db_session, tmp_path, _big_text(), name="big.md")
        outcome = await _extract(db_session, entry_id, snapshot_id, _Scripted(["refuse", _REPLY]))
        snap = await db_session.get(SnapshotRow, snapshot_id)
        assert snap is not None and snap.extraction_status == ExtractionStatus.PENDING.value
        assert outcome.refused_calls == 1
        assert not outcome.refusal_recorded
        assert await _rows(db_session) == []

    @pytest.mark.asyncio
    async def test_case_11_every_chunk_refused_is_recorded(
        self, db_session: AsyncSession, tmp_path: Path
    ) -> None:
        entry_id, snapshot_id = await _deposit(db_session, tmp_path, _big_text(), name="big.md")
        outcome = await _extract(db_session, entry_id, snapshot_id, _Scripted(["refuse"]))
        assert outcome.refused_calls > 1
        assert outcome.refusal_recorded
        assert len(await _rows(db_session)) == 1

    @pytest.mark.asyncio
    async def test_case_14_append_only_entry_is_never_recorded(
        self, db_session: AsyncSession, tmp_path: Path
    ) -> None:
        entry_id, snapshot_id = await _deposit(
            db_session, tmp_path, mutability=Mutability.APPEND_ONLY, name="log.md"
        )
        outcome = await _extract(db_session, entry_id, snapshot_id, _Scripted(["refuse"]))
        assert outcome.refused_calls == 1
        assert not outcome.refusal_recorded
        assert await _rows(db_session) == []

    @pytest.mark.asyncio
    async def test_case_5_a_later_success_clears_the_record(
        self, db_session: AsyncSession, tmp_path: Path, mock_encoder: None
    ) -> None:
        entry_id, snapshot_id = await _deposit(db_session, tmp_path)
        await _extract(db_session, entry_id, snapshot_id, _Scripted(["refuse"]))
        await _extract(db_session, entry_id, snapshot_id, _Scripted([_REPLY]))
        snap = await db_session.get(SnapshotRow, snapshot_id)
        assert snap is not None and snap.extraction_status == ExtractionStatus.COMPLETE.value
        assert await _rows(db_session) == []

    @pytest.mark.asyncio
    async def test_a_second_refusal_refreshes_the_row(
        self, db_session: AsyncSession, tmp_path: Path
    ) -> None:
        entry_id, snapshot_id = await _deposit(db_session, tmp_path)
        await _extract(db_session, entry_id, snapshot_id, _Scripted(["refuse"]))
        await _extract(db_session, entry_id, snapshot_id, _Scripted(["refuse"]))
        [row] = await _rows(db_session)
        assert row.refusals == 2


# ---------------------------------------------------------------------------
# §4: the hold
# ---------------------------------------------------------------------------


class TestHold:
    @pytest.mark.asyncio
    async def test_case_2_a_recorded_refusal_is_held(
        self, db_session: AsyncSession, tmp_path: Path
    ) -> None:
        entry_id, snapshot_id = await _deposit(db_session, tmp_path)
        await _extract(db_session, entry_id, snapshot_id, _Scripted(["refuse"]))
        [held] = await held_snapshots(db_session)
        assert held.snapshot_id == snapshot_id and held.category == "cyber"

    @pytest.mark.asyncio
    async def test_case_6_a_model_change_releases_the_hold(
        self, db_session: AsyncSession, tmp_path: Path
    ) -> None:
        entry_id, snapshot_id = await _deposit(db_session, tmp_path)
        await _extract(db_session, entry_id, snapshot_id, _Scripted(["refuse"]))
        other = await list_held_snapshots(db_session, {"extraction": "anthropic:other"}, 30)
        assert other == []

    @pytest.mark.asyncio
    async def test_case_8_the_hold_lapses_after_the_retry_window(
        self, db_session: AsyncSession, tmp_path: Path
    ) -> None:
        entry_id, snapshot_id = await _deposit(db_session, tmp_path)
        await _extract(db_session, entry_id, snapshot_id, _Scripted(["refuse"]))
        pairings = current_refusal_pairings()
        later = datetime.now(UTC) + timedelta(days=31)
        assert await list_held_snapshots(db_session, pairings, 30, now=later) == []
        assert len(await list_held_snapshots(db_session, pairings, 30)) == 1
        assert await list_held_snapshots(db_session, pairings, 0) == []

    @pytest.mark.asyncio
    async def test_case_7_an_edit_is_never_held(
        self, db_session: AsyncSession, tmp_path: Path
    ) -> None:
        entry_id, snapshot_id = await _deposit(db_session, tmp_path)
        await _extract(db_session, entry_id, snapshot_id, _Scripted(["refuse"]))
        _, edited = await _deposit(db_session, tmp_path, "LUKS notes, edited.", name="luks2.md")
        held = {h.snapshot_id for h in await held_snapshots(db_session)}
        assert snapshot_id in held and edited not in held

    @pytest.mark.asyncio
    async def test_catchup_listing_leaves_held_snapshots_out(
        self, db_session: AsyncSession, tmp_path: Path
    ) -> None:
        from particles.operations.consolidation import catchup_listing

        entry_id, snapshot_id = await _deposit(db_session, tmp_path)
        _, other = await _deposit(db_session, tmp_path, "Another note.", name="other.md")
        await _extract(db_session, entry_id, snapshot_id, _Scripted(["refuse"]))
        pending, held = await catchup_listing(db_session)
        assert [p.snapshot_id for p in pending] == [other]
        assert held == 1


# ---------------------------------------------------------------------------
# §5: the surfaces
# ---------------------------------------------------------------------------


class TestSurfaces:
    @pytest.mark.asyncio
    async def test_case_15_curate_card_names_the_source(
        self, db_session: AsyncSession, tmp_path: Path
    ) -> None:
        from particles.operations.curation.cards import CardKind, CurationCard
        from particles.operations.curation.collect import refused_snapshots_card

        entry_id, snapshot_id = await _deposit(db_session, tmp_path, tags=["project:alpha"])
        await _extract(db_session, entry_id, snapshot_id, _Scripted(["refuse"]))
        card = await refused_snapshots_card(db_session)
        assert card is not None and card.kind is CardKind.REFUSED_SNAPSHOTS
        assert "luks.md (cyber; in view for alpha)" in card.diagnostic
        assert card.key == "refused_snapshots"
        assert CurationCard.from_key(card.key).kind is CardKind.REFUSED_SNAPSHOTS
        assert card.suggested_gestures == ["retry-refused", "snooze"]

    @pytest.mark.asyncio
    async def test_case_18_no_refusal_no_card(self, db_session: AsyncSession) -> None:
        from particles.operations.curation.collect import refused_snapshots_card

        assert await refused_snapshots_card(db_session) is None

    @pytest.mark.asyncio
    async def test_case_16_digest_count_follows_observer_scope(
        self, db_session: AsyncSession, tmp_path: Path
    ) -> None:
        from particles.operations.digest import _refused_in_view

        entry_id, snapshot_id = await _deposit(db_session, tmp_path, tags=["project:alpha"])
        await _extract(db_session, entry_id, snapshot_id, _Scripted(["refuse"]))
        assert await _refused_in_view(db_session, None) == 1
        assert await _refused_in_view(db_session, "alpha") == 1
        assert await _refused_in_view(db_session, "beta") == 0

    def test_digest_line_renders_and_absence_is_byte_identical(self) -> None:
        from particles.render.markdown import render_digest_located

        plain, _ = render_digest_located("default", [], 0)
        assert render_digest_located("default", [], 0, refused_line=None)[0] == plain
        text, _ = render_digest_located(
            "default", [], 0, refused_line="1 source(s) are not in memory"
        )
        assert "_1 source(s) are not in memory_" in text

    @pytest.mark.asyncio
    async def test_case_17_quality_counts_held(
        self, db_session: AsyncSession, tmp_path: Path
    ) -> None:
        from particles.operations.quality import get_quality_report

        entry_id, snapshot_id = await _deposit(db_session, tmp_path)
        await _extract(db_session, entry_id, snapshot_id, _Scripted(["refuse"]))
        report = await get_quality_report(db_session)
        assert report.snapshots_pending == 1
        assert report.snapshots_held_refused == 1

    def test_audit_lines_report_held_apart_from_failures(self) -> None:
        from particles.operations.audit import AuditReport, _extraction_lines

        report = AuditReport(
            store="default", extracted_snapshots=2, extraction_held_files=["luks.md"]
        )
        lines = _extraction_lines(report)
        assert "all 2 files extracted in full" in lines[0]
        assert "held: refused" in lines[1] and "luks.md" in lines[1]
        assert report.complete

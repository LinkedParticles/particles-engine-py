# SPDX-FileCopyrightText: 2026 The Particles authors
# SPDX-License-Identifier: Apache-2.0

"""Real Engine-boundary regressions; synthetic text, no models or network."""

import copy
import hashlib
import json
from pathlib import Path

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from particles.core.schema import Mutability
from particles.corpus.deposit import ForeignEntryRefused, deposit_text_versioned
from particles.corpus.harvest_marks import HARVEST_DEPOSITORS, RESERVED_DEPOSITORS, RESERVED_TAGS
from particles.corpus.store import CorpusEntryRow, SnapshotRow
from particles.extraction.codex_transcript import Ancestry, UnsupportedTranscript
from particles.operations.codex_capture import capture_codex
from particles.store.codex_capture_store import CodexCaptureRow, get_capture
from tests.test_codex_transcript import SID, blob, message, metadata

HOME_ID = "30000000-0000-4000-8000-000000000003"


def raw_source(identity: str = SID, text: str = "Synthetic preference: blue") -> bytes:
    meta = metadata()
    meta["payload"]["id"] = identity
    return blob(meta, message(text)).encode()


async def capture(session: AsyncSession, home: Path, raw: bytes, **kwargs: object):
    return await capture_codex(
        session,
        raw=raw,
        home=home,
        home_id=HOME_ID,
        project_key="test-project",
        launch_root=Path("/tmp/repository"),
        **kwargs,
    )


@pytest.mark.asyncio
async def test_atomic_noop_append_and_namespace_hash_isolation(
    db_session: AsyncSession, tmp_path: Path
) -> None:
    initial = raw_source()
    first, unchanged = await capture(db_session, tmp_path, initial)
    assert not unchanged
    from particles.corpus.harvest_marks import HARNESS_SOURCES

    row = await db_session.get(CorpusEntryRow, first.entry_id)
    source = HARNESS_SOURCES["codex"]
    assert row.uri_r == source.session_prefix + SID
    assert row.deposited_by == source.depositor and source.tag in json.loads(row.tags_json)
    repeat, unchanged = await capture(db_session, tmp_path, initial)
    assert unchanged and repeat.capture_id == first.capture_id
    added = initial + blob(message("New synthetic claim", "second")).encode()
    grown, unchanged = await capture(db_session, tmp_path, added)
    assert (
        not unchanged
        and grown.entry_id == first.entry_id
        and grown.snapshot_id != first.snapshot_id
    )
    other, _ = await capture(
        db_session, tmp_path, raw_source("40000000-0000-4000-8000-000000000004")
    )
    assert other.entry_id != first.entry_id
    assert (await db_session.scalar(select(func.count()).select_from(CorpusEntryRow))) == 2
    assert (await db_session.scalar(select(func.count()).select_from(SnapshotRow))) == 3


@pytest.mark.parametrize("depositor", ["operator", "claude-code-hook", "audit"])
@pytest.mark.asyncio
async def test_same_bytes_foreign_uri_is_refused_before_noop_or_tag_reconcile(
    db_session: AsyncSession, tmp_path: Path, depositor: str
) -> None:
    from particles.corpus.redaction import redact_secrets
    from particles.extraction.codex_transcript import parse_transcript

    text, _ = parse_transcript(raw_source().decode()).render(redact_secrets)
    entry_id, _, _ = await deposit_text_versioned(
        db_session,
        text=text,
        uri_r=f"codex://session/{SID}",
        source_type="CONVERSATION",
        mutability=Mutability.APPEND_ONLY,
        deposited_by=depositor,
        tags=["project:original"],
    )
    with pytest.raises(ForeignEntryRefused):
        await capture(db_session, tmp_path, raw_source())
    entry = await db_session.get(CorpusEntryRow, entry_id)
    assert entry and json.loads(entry.tags_json) == ["project:original"]
    assert await get_capture(db_session, SID) is None


@pytest.mark.asyncio
async def test_cross_scheme_identity_and_nonappend_history_refuse(
    db_session: AsyncSession, tmp_path: Path
) -> None:
    bad = metadata()
    bad["payload"]["id"] = "file:///tmp/claude"
    with pytest.raises(UnsupportedTranscript, match="invalid_session"):
        await capture(db_session, tmp_path, blob(bad, message("x")).encode())
    good, _ = await capture(db_session, tmp_path, raw_source())
    with pytest.raises(UnsupportedTranscript, match="non_append_history"):
        await capture(db_session, tmp_path, raw_source(text="replaced history"))
    assert (await get_capture(db_session, SID)).capture_id == good.capture_id


@pytest.mark.asyncio
async def test_moved_home_refuses_even_identical_bytes(
    db_session: AsyncSession, tmp_path: Path
) -> None:
    await capture(db_session, tmp_path / "original", raw_source())
    with pytest.raises(UnsupportedTranscript, match="codex_home_moved"):
        await capture(db_session, tmp_path / "restored", raw_source())


@pytest.mark.asyncio
async def test_raw_growth_without_new_testimony_does_not_queue_extraction(
    db_session: AsyncSession, tmp_path: Path
) -> None:
    initial = raw_source()
    old, _ = await capture(db_session, tmp_path, initial)
    more = (
        initial
        + blob(
            {"type": "event_msg", "payload": {"type": "agent_message", "message": "mirror"}}
        ).encode()
    )
    new, unchanged = await capture(db_session, tmp_path, more)
    assert unchanged and new.snapshot_id == old.snapshot_id and new.capture_id != old.capture_id
    assert (await db_session.scalar(select(func.count()).select_from(SnapshotRow))) == 1


@pytest.mark.asyncio
async def test_fork_requires_engine_verified_parent_prefix(
    db_session: AsyncSession, tmp_path: Path
) -> None:
    records = [metadata(), message("parent")]
    for ordinal, record in enumerate(records):
        record["ordinal"] = ordinal
    parent_raw = blob(*records).encode()
    boundary = Ancestry(SID, len(records), len(parent_raw))
    child = copy.deepcopy(metadata())
    child["payload"]["id"] = "40000000-0000-4000-8000-000000000004"
    child["payload"]["history_base"] = {
        "thread_id": SID,
        "end_ordinal_exclusive": len(records),
        "end_byte_offset": len(parent_raw),
    }
    child_raw = blob(child, message("child only")).encode()
    with pytest.raises(UnsupportedTranscript, match="unverified_fork"):
        await capture(db_session, tmp_path, child_raw)
    parent, _ = await capture(db_session, tmp_path, parent_raw, requested_prefix=boundary)
    captured, _ = await capture(db_session, tmp_path, child_raw)
    assert captured.parent and captured.parent["sha256"] == hashlib.sha256(parent_raw).hexdigest()
    assert len(captured.messages) == 1
    assert parent.session_id == SID


@pytest.mark.asyncio
async def test_redaction_precedes_body_proof(db_session: AsyncSession, tmp_path: Path) -> None:
    secret = "AKIA" + "X" * 16
    manifest, _ = await capture(db_session, tmp_path, raw_source(text=secret))
    assert secret not in manifest.model_dump_json()
    assert manifest.messages[0].body_hash == hashlib.sha256(b"[REDACTED AWS KEY]").hexdigest()
    stored = await db_session.get(CodexCaptureRow, manifest.capture_id)
    assert stored and secret not in stored.payload_json


def test_codex_does_not_join_claude_shared_ownership() -> None:
    assert "codex-hook" in RESERVED_DEPOSITORS
    assert "codex-hook" not in HARVEST_DEPOSITORS
    assert "codex" in RESERVED_TAGS


@pytest.mark.asyncio
async def test_codex_capture_is_invisible_to_utility_mining(db_session, tmp_path):
    from particles.operations.utility_mining import harvested_sessions

    await capture(db_session, tmp_path, raw_source())
    sessions, skipped = await harvested_sessions(db_session)
    assert sessions == [] and skipped == 0


@pytest.mark.asyncio
async def test_empty_session_has_committed_provenance_and_cannot_mask_a_prefix_removal(
    db_session, tmp_path
):
    empty = blob(metadata()).encode()
    manifest, _ = await capture(db_session, tmp_path, empty)
    assert manifest.text_size == 0 and manifest.messages == []
    await db_session.commit()
    grown, _ = await capture(db_session, tmp_path, raw_source())
    await db_session.commit()
    with pytest.raises(UnsupportedTranscript, match="non_append_history"):
        await capture(db_session, tmp_path, empty)
    assert (await get_capture(db_session, SID)).snapshot_id == grown.snapshot_id


@pytest.mark.asyncio
async def test_empty_capture_extracts_without_a_model_call(db_session, tmp_path, monkeypatch):
    from unittest.mock import AsyncMock

    from particles.corpus.store import get_snapshot
    from particles.extraction import general

    called = AsyncMock(side_effect=AssertionError("empty source must not call a model"))
    monkeypatch.setattr(general, "_call_llm", called)
    manifest, _ = await capture(db_session, tmp_path, blob(metadata()).encode())
    snapshot = await get_snapshot(db_session, manifest.snapshot_id)
    result = await general.GeneralExtractor().extract(snapshot, b"", source_type="CONVERSATION")
    assert result.candidates == []
    assert any("Empty content" in note for note in result.quality_notes)
    called.assert_not_called()


@pytest.mark.asyncio
async def test_identical_content_never_joins_a_foreign_scheme_or_depositor(db_session, tmp_path):
    from particles.corpus.redaction import redact_secrets
    from particles.extraction.codex_transcript import parse_transcript

    text, _ = parse_transcript(raw_source().decode()).render(redact_secrets)
    foreign, _, _ = await deposit_text_versioned(
        db_session,
        text=text,
        uri_r="file:///synthetic/claude-session.jsonl",
        source_type="CONVERSATION",
        mutability=Mutability.APPEND_ONLY,
        deposited_by="claude-code-hook",
        tags=["project:foreign"],
    )
    captured, _ = await capture(db_session, tmp_path, raw_source())
    assert captured.entry_id != foreign
    row = await db_session.get(CorpusEntryRow, foreign)
    assert row.uri_r == "file:///synthetic/claude-session.jsonl"
    assert row.deposited_by == "claude-code-hook"
    assert json.loads(row.tags_json) == ["project:foreign"]
    assert (await db_session.scalar(select(func.count()).select_from(CorpusEntryRow))) == 2

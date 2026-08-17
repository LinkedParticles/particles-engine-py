"""Tests for ``deposit_text_versioned``.

The versioned text deposit is the Claude Code harvest's engine-side seam:
identity is the caller-supplied URI-R (one corpus entry per logical source),
an unchanged re-deposit is a no-op, and a changed re-deposit appends a
snapshot under the entry's mutability semantics.
"""

from __future__ import annotations

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from particles.core.schema import Mutability
from particles.corpus.deposit import deposit_text_versioned
from particles.corpus.store import get_entry_by_uri, list_snapshots_for_entry

URI = "claude-code://session/test-session-1"


@pytest.mark.asyncio
async def test_initial_deposit_creates_entry_and_snapshot(db_session: AsyncSession) -> None:
    entry_id, snapshot_id, unchanged = await deposit_text_versioned(
        db_session,
        text="# Session\n\n## User\n\nhello\n",
        uri_r=URI,
        source_type="CONVERSATION",
        mutability=Mutability.APPEND_ONLY,
        tags=["claude-code", "session:test-session-1"],
        deposited_by="claude-code-hook",
    )
    assert not unchanged
    entry = await get_entry_by_uri(db_session, URI)
    assert entry is not None
    assert entry.entry_id == entry_id
    assert entry.source_type == "CONVERSATION"
    assert entry.mutability == Mutability.APPEND_ONLY
    assert "claude-code" in entry.tags
    snapshots = await list_snapshots_for_entry(db_session, entry_id)
    assert [s.snapshot_id for s in snapshots] == [snapshot_id]


@pytest.mark.asyncio
async def test_unchanged_redeposit_is_noop(db_session: AsyncSession) -> None:
    text = "# Session\n\n## User\n\nhello\n"
    entry_id, snapshot_id, _ = await deposit_text_versioned(
        db_session,
        text=text,
        uri_r=URI,
        source_type="CONVERSATION",
        mutability=Mutability.APPEND_ONLY,
    )
    entry_id2, snapshot_id2, unchanged = await deposit_text_versioned(
        db_session,
        text=text,
        uri_r=URI,
        source_type="CONVERSATION",
        mutability=Mutability.APPEND_ONLY,
    )
    assert unchanged
    assert (entry_id2, snapshot_id2) == (entry_id, snapshot_id)
    assert len(await list_snapshots_for_entry(db_session, entry_id)) == 1


@pytest.mark.asyncio
async def test_grown_content_appends_snapshot_to_same_entry(db_session: AsyncSession) -> None:
    base = "# Session\n\n## User\n\nhello\n"
    entry_id, _, _ = await deposit_text_versioned(
        db_session,
        text=base,
        uri_r=URI,
        source_type="CONVERSATION",
        mutability=Mutability.APPEND_ONLY,
    )
    entry_id2, snapshot_id2, unchanged = await deposit_text_versioned(
        db_session,
        text=base + "\n## Assistant\n\nhi\n",
        uri_r=URI,
        source_type="CONVERSATION",
        mutability=Mutability.APPEND_ONLY,
    )
    assert not unchanged
    assert entry_id2 == entry_id  # one corpus entry per logical source
    snapshots = await list_snapshots_for_entry(db_session, entry_id)
    assert len(snapshots) == 2
    assert snapshot_id2 in {s.snapshot_id for s in snapshots}

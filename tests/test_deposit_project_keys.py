# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Project keys on re-deposit — additive, and only among attributed entries.

A keyed ``MUTABLE`` deposit never shares another URI's entry by content hash:
two projects' identical memory files are two sources.

Entry tags were written once, when the row was created, which froze an entry's
project at whoever deposited it first. These tests pin the reconcile rule and
its one refusal: a project's deposit never rides an operator's global entry.
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from particles.core.schema import Mutability
from particles.corpus.deposit import deposit_text, deposit_text_versioned
from particles.corpus.store import get_entry


async def _versioned(session: AsyncSession, uri: str, text: str, tags: list[str]) -> str:
    entry_id, _snapshot_id, _unchanged = await deposit_text_versioned(
        session,
        text=text,
        uri_r=uri,
        source_type="CONVERSATION",
        mutability=Mutability.APPEND_ONLY,
        tags=tags,
    )
    return entry_id


async def _tags(session: AsyncSession, entry_id: str) -> list[str]:
    entry = await get_entry(session, entry_id)
    assert entry is not None
    return entry.tags


async def test_an_audit_first_transcript_gains_its_key_when_the_hook_sees_it(
    db_session: AsyncSession,
) -> None:
    uri = "claude-code://session/s1"
    first = await _versioned(db_session, uri, "turn one", ["claude-code", "audit"])

    # Same session, unchanged bytes, now deposited with the project known.
    again = await _versioned(db_session, uri, "turn one", ["claude-code", "project:-a-repo"])

    assert again == first
    assert await _tags(db_session, first) == ["claude-code", "audit", "project:-a-repo"]


async def test_a_second_projects_key_is_added_and_nothing_is_removed(
    db_session: AsyncSession,
) -> None:
    uri = "file:///shared/notes.md"
    entry_id = await _versioned(db_session, uri, "v1", ["claude-code", "project:-a"])

    await _versioned(db_session, uri, "v2", ["claude-code", "project:-b"])
    await _versioned(db_session, uri, "v2", ["claude-code", "project:-b"])  # idempotent

    assert await _tags(db_session, entry_id) == ["claude-code", "project:-a", "project:-b"]


async def test_a_key_is_never_added_to_a_global_entry(db_session: AsyncSession) -> None:
    uri = "file:///home/me/handbook.md"
    entry_id = await _versioned(db_session, uri, "the handbook", ["handbook"])

    await _versioned(db_session, uri, "the handbook", ["claude-code", "project:-a"])

    assert await _tags(db_session, entry_id) == ["handbook"]


async def _memory_file(session: AsyncSession, uri: str, text: str, key: str) -> tuple[str, str]:
    entry_id, snapshot_id, _unchanged = await deposit_text_versioned(
        session,
        text=text,
        uri_r=uri,
        source_type="LOCAL_MARKDOWN",
        mutability=Mutability.MUTABLE,
        tags=["claude-code", "memory-file", f"project:{key}"],
    )
    return entry_id, snapshot_id


async def test_byte_identical_memory_files_from_two_projects_are_two_entries(
    db_session: AsyncSession,
) -> None:
    """A keyed MUTABLE deposit is identified by its URI alone.

    Sharing one entry would give one project's later edit both projects' keys.
    """
    first, _ = await _memory_file(db_session, "file:///a/memory/x.md", "same", "-a")
    second, _ = await _memory_file(db_session, "file:///b/memory/x.md", "same", "-b")

    assert second != first
    assert await _tags(db_session, first) == ["claude-code", "memory-file", "project:-a"]
    assert await _tags(db_session, second) == ["claude-code", "memory-file", "project:-b"]


async def test_an_unchanged_memory_file_never_matches_another_projects_old_snapshot(
    db_session: AsyncSession,
) -> None:
    """The second failure the shared entry caused."""
    a, _ = await _memory_file(db_session, "file:///a/memory/x.md", "v1", "-a")
    await _memory_file(db_session, "file:///a/memory/x.md", "v2", "-a")

    b, b_snapshot = await _memory_file(db_session, "file:///b/memory/x.md", "v1", "-b")

    assert b != a
    b_entry = await get_entry(db_session, b)
    assert b_entry is not None
    assert [s.snapshot_id for s in b_entry.snapshots] == [b_snapshot]


async def test_byte_identical_append_only_documents_still_share_an_entry_and_both_keys(
    db_session: AsyncSession,
) -> None:
    """Only a keyed MUTABLE deposit leaves content-hash dedup."""
    first = await _versioned(db_session, "file:///a/memory/x.md", "same", ["project:-a"])
    second = await _versioned(db_session, "file:///b/memory/x.md", "same", ["project:-b"])

    assert second == first  # content-hash dedup, as before
    assert await _tags(db_session, first) == ["project:-a", "project:-b"]


async def test_a_keyed_deposit_does_not_attach_to_a_global_entry_by_content_hash(
    db_session: AsyncSession,
) -> None:
    """Otherwise an agent's excerpt matching an operator's page would read as global."""
    page, _ = await deposit_text(db_session, "a public fact", tags=["web"])

    excerpt, _ = await deposit_text(db_session, "a public fact", tags=["claude-code", "project:-a"])

    assert excerpt != page
    assert await _tags(db_session, page) == ["web"]
    assert await _tags(db_session, excerpt) == ["claude-code", "project:-a"]


async def test_a_keyless_deposit_still_dedups_onto_a_global_entry(db_session: AsyncSession) -> None:
    page, _ = await deposit_text(db_session, "a public fact", tags=["web"])
    again, _ = await deposit_text(db_session, "a public fact", tags=["web"])
    assert again == page

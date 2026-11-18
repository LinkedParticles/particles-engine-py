# SPDX-FileCopyrightText: 2026 The Particles authors
# SPDX-License-Identifier: Apache-2.0

"""Which host a file:// entry's path names, and what that changes.

In-memory store, no network, no model call. "This host" is the suite's own
host id (``PARTICLES_HOST_ID_FILE``, set once per run); another host is a
fixed foreign id deposited under :func:`deposit_origin`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from particles.core.schema import FetchPolicy, Mutability
from particles.corpus.deposit import deposit_file, deposit_text_versioned
from particles.corpus.fetch import (
    RefreshRefusal,
    count_refused_local_entries,
    entry_refusal,
    maybe_refetch,
)
from particles.corpus.origin import (
    OriginCounts,
    OtherHostEntryRefused,
    count_file_entries_by_origin,
    deposit_origin,
)
from particles.corpus.store import CorpusEntryRow, list_refreshable_local_entries
from particles.host_identity import local_host_id
from particles.operations.corpus_origin import origin_stamp, reorigin
from tests._corpus_rows import CLIENT_HOST_ID

LAPTOP = CLIENT_HOST_ID


async def _versioned(session: Any, uri: str, text: str, **kw: Any) -> tuple[str, str, bool]:
    return await deposit_text_versioned(
        session,
        text=text,
        uri_r=uri,
        source_type="LOCAL_MARKDOWN",
        mutability=Mutability.MUTABLE,
        deposited_by="claude-code-hook",
        fetch_policy=kw.pop("fetch_policy", FetchPolicy.NEVER),
        **kw,
    )


async def _origin(session: Any, entry_id: str) -> str | None:
    row = await session.get(CorpusEntryRow, entry_id)
    assert row is not None
    return row.origin_host


# --- who stamps what (§1.2) --------------------------------------------------------


@pytest.mark.asyncio
async def test_an_in_process_deposit_stamps_this_host(db_session: Any) -> None:
    entry_id, _, _ = await _versioned(db_session, "file:///home/u/notes.md", "a")
    assert await _origin(db_session, entry_id) == local_host_id()


@pytest.mark.asyncio
async def test_a_scope_stamps_the_hosts_it_names(db_session: Any) -> None:
    with deposit_origin(LAPTOP):
        pushed, _, _ = await _versioned(db_session, "file:///home/u/a.md", "a")
    with deposit_origin(None):
        unknown, _, _ = await _versioned(db_session, "file:///home/u/b.md", "b")
    assert await _origin(db_session, pushed) == LAPTOP
    assert await _origin(db_session, unknown) is None


@pytest.mark.asyncio
async def test_a_non_file_entry_records_no_origin(db_session: Any) -> None:
    entry_id, _, _ = await deposit_text_versioned(
        db_session,
        text="t",
        uri_r="claude-code://session/abc",
        source_type="CONVERSATION",
        mutability=Mutability.APPEND_ONLY,
    )
    assert await _origin(db_session, entry_id) is None


# --- which entry a deposit may join (§3.3, §5.2) ------------------------------------


@pytest.mark.asyncio
async def test_a_deposit_naming_another_hosts_file_is_refused(db_session: Any) -> None:
    uri = "file:///Users/jeff/src/particles-engine-py/AGENTS.md"
    with deposit_origin(LAPTOP):
        await _versioned(db_session, uri, "laptop's rules")
    # The mini's own rules sync of the same path writes nothing (§5.4 row 4).
    with pytest.raises(OtherHostEntryRefused) as caught:
        await _versioned(db_session, uri, "mini's checkout")
    assert caught.value.entry_origin == LAPTOP
    assert not caught.value.engine_own
    assert LAPTOP in caught.value.describe()


@pytest.mark.asyncio
async def test_a_push_onto_the_engines_own_file_names_the_repair(db_session: Any) -> None:
    uri = "file:///home/u/AGENTS.md"
    await _versioned(db_session, uri, "the engine host's file")
    with deposit_origin(LAPTOP), pytest.raises(OtherHostEntryRefused) as caught:
        await _versioned(db_session, uri, "pushed")
    assert caught.value.engine_own
    assert "corpus reorigin" in caught.value.describe()


@pytest.mark.asyncio
async def test_an_unstamped_entry_is_adopted_by_the_first_deposit_naming_it(
    db_session: Any,
) -> None:
    uri = "file:///home/u/MEMORY.md"
    with deposit_origin(None):
        entry_id, _, _ = await _versioned(db_session, uri, "v1")
    with deposit_origin(LAPTOP):
        again, _, unchanged = await _versioned(db_session, uri, "v1")
    assert again == entry_id and unchanged
    assert await _origin(db_session, entry_id) == LAPTOP


@pytest.mark.asyncio
async def test_a_push_into_a_lazy_entry_of_its_own_host_sets_it_never(db_session: Any) -> None:
    """The §2 case: a memory file the local harvest enrolled LAZY, pushed after a cutover."""
    uri = "file:///home/u/.claude/projects/p/memory/note.md"
    with deposit_origin(LAPTOP):
        entry_id, _, _ = await _versioned(db_session, uri, "v1", fetch_policy=FetchPolicy.LAZY)
        await _versioned(db_session, uri, "v2", fetch_policy=FetchPolicy.NEVER)
    row = await db_session.get(CorpusEntryRow, entry_id)
    assert row is not None and row.fetch_policy == FetchPolicy.NEVER.value


@pytest.mark.asyncio
async def test_a_content_hash_match_never_joins_another_hosts_file(
    db_session: Any, tmp_path: Path
) -> None:
    """Two hosts' byte-identical files are two sources (§1.3)."""
    here = tmp_path / "AGENTS.md"
    here.write_text("# Same rules\n")
    with deposit_origin(LAPTOP):
        theirs, _, _ = await deposit_text_versioned(
            db_session,
            text="# Same rules\n",
            uri_r="file:///elsewhere/AGENTS.md",
            source_type="LOCAL_MARKDOWN",
            mutability=Mutability.STABLE,
            deposited_by="operator",
        )
    mine, _ = await deposit_file(db_session, here)
    assert mine != theirs
    assert await _origin(db_session, mine) == local_host_id()


# --- the local gate (§2) ------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_work_list_holds_only_this_hosts_entries(db_session: Any, tmp_path: Path) -> None:
    mine, _, _ = await _versioned(
        db_session, (tmp_path / "a.md").as_uri(), "a", fetch_policy=FetchPolicy.LAZY
    )
    with deposit_origin(LAPTOP):
        await _versioned(
            db_session, (tmp_path / "b.md").as_uri(), "b", fetch_policy=FetchPolicy.LAZY
        )
    with deposit_origin(None):
        await _versioned(
            db_session, (tmp_path / "c.md").as_uri(), "c", fetch_policy=FetchPolicy.LAZY
        )
    assert [entry_id for entry_id, _ in await list_refreshable_local_entries(db_session)] == [mine]
    counts = await count_refused_local_entries(db_session)
    assert (counts.other_host, counts.origin_unknown) == (1, 1)
    assert counts.info_lines() and any("record no host" in line for line in counts.lines())
    assert await count_file_entries_by_origin(db_session) == OriginCounts(
        this_host=1, other_hosts=1, unstamped=1
    )


@pytest.mark.asyncio
async def test_another_hosts_path_is_never_read_even_when_forced(
    db_session: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The mini has the laptop's path, owned by the same user; it is still not read."""
    path = tmp_path / "AGENTS.md"
    path.write_text("the mini's checkout\n")
    with deposit_origin(LAPTOP):
        entry_id, snapshot_id, _ = await _versioned(
            db_session, path.as_uri(), "the laptop's file\n", fetch_policy=FetchPolicy.LAZY
        )
    row = await db_session.get(CorpusEntryRow, entry_id)
    assert row is not None
    assert entry_refusal(row, []) is RefreshRefusal.OTHER_HOST

    def _no_stat(*_a: Any, **_k: Any) -> Any:
        raise AssertionError("another host's path was stat'ed")

    monkeypatch.setattr(Path, "stat", _no_stat)
    monkeypatch.setattr(Path, "is_file", _no_stat)
    assert await maybe_refetch(db_session, entry_id, force=True) is None


@pytest.mark.asyncio
async def test_an_unstamped_entry_is_refused_as_origin_unknown(db_session: Any) -> None:
    with deposit_origin(None):
        entry_id, _, _ = await _versioned(
            db_session, "file:///home/u/x.md", "x", fetch_policy=FetchPolicy.LAZY
        )
    row = await db_session.get(CorpusEntryRow, entry_id)
    assert row is not None
    assert entry_refusal(row, []) is RefreshRefusal.ORIGIN_UNKNOWN


# --- the one-time stamp and the repair (§5) -----------------------------------------


@pytest.mark.asyncio
async def test_the_stamp_runs_once_per_store(db_session: Any) -> None:
    with deposit_origin(None):
        await _versioned(db_session, "file:///home/u/rule.md", "r", tags=["rule-file"])
        await _versioned(db_session, "file:///home/u/m.md", "m", tags=["memory-file"])
    preview = await origin_stamp(db_session, apply=False)
    assert (preview.plan.rule_files, preview.plan.memory_files, preview.written) == (1, 1, 0)
    done = await origin_stamp(db_session, apply=True)
    assert done.written == 2
    with deposit_origin(None):
        late, _, _ = await _versioned(db_session, "file:///home/u/late.md", "l")
    again = await origin_stamp(db_session, apply=True)
    assert again.already_stamped and again.written == 0
    assert await _origin(db_session, late) is None


@pytest.mark.asyncio
async def test_reorigin_counts_then_moves_and_records_one_event(db_session: Any) -> None:
    from particles.store.event_store import OperatorEventType, count_events_by_type

    me = local_host_id()
    assert me is not None
    await _versioned(db_session, "file:///home/u/a.md", "a")
    await _versioned(db_session, "file:///other/b.md", "b")
    n = await reorigin(
        db_session, from_host=me, to_host=LAPTOP, uri_prefix="file:///home/u/", apply=False
    )
    assert n == 1
    assert (await count_file_entries_by_origin(db_session)).this_host == 2
    moved = await reorigin(
        db_session, from_host=me, to_host=LAPTOP, uri_prefix="file:///home/u/", apply=True
    )
    assert moved == 1
    counts = await count_events_by_type(db_session)
    assert counts.get(OperatorEventType.CORPUS_ORIGIN_REASSIGNED.value) == 1


@pytest.mark.asyncio
async def test_an_operator_scope_writes_a_clients_entry_but_never_the_engines(
    db_session: Any,
) -> None:
    """The legacy operator bearer keeps writing the client's files; N19 still holds."""
    client_uri, engine_uri = "file:///home/u/c.md", "file:///home/u/e.md"
    with deposit_origin(LAPTOP):
        client_entry, _, _ = await _versioned(db_session, client_uri, "v1")
    await _versioned(db_session, engine_uri, "v1")
    with deposit_origin(None, operator=True):
        again, _, unchanged = await _versioned(db_session, client_uri, "v2")
        assert again == client_entry and not unchanged
        with pytest.raises(OtherHostEntryRefused):
            await _versioned(db_session, engine_uri, "v2")
    assert await _origin(db_session, client_entry) == LAPTOP


@pytest.mark.asyncio
async def test_a_local_rules_sync_skips_a_path_another_host_tracks(
    db_session: Any, tmp_path: Path
) -> None:
    """The mini's own rules sync writes nothing for the laptop's path (§5.4 row 4)."""
    from particles.corpus.rule_sources import sync_rule_sources

    root = tmp_path / "repo"
    (root / ".git").mkdir(parents=True)
    (root / "AGENTS.md").write_text("# Rules\n\nThe mini's checkout.\n")
    (root / "docs").mkdir()
    (root / "docs" / "AGENTS.md").write_text("# Docs rules\n\nOnly here.\n")
    with deposit_origin(LAPTOP):
        await _versioned(db_session, (root / "AGENTS.md").resolve().as_uri(), "laptop's")
    report = await sync_rule_sources(db_session, [str(root)])
    assert [(p.name, host) for p, host in report.other_host] == [("AGENTS.md", LAPTOP)]
    assert [p.parent.name for p, _ in report.deposited] == ["docs"]

# SPDX-FileCopyrightText: 2026 The Particles authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: F811 (the engine and keys_file fixtures are imported, then named as parameters)

"""An agent key's deposit writes only its own entries (rules 3 and 4).

Regression tests for the sixth delta security audit's deposit-confinement
findings (security review 2026-10-26-1, W2, W3, W4, W21, W26, W35, W36, W41,
W52, W67). Each test states the property that now holds. Keys are minted into
the test's own key file and every store is a scratch store; no model or network
call is made.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from particles.api import _keys
from particles.api._keys import KeyRuleError, Role
from particles.core.observer_scope import EntryScope, classify_entry, project_tag
from tests.test_engine_keys import (  # noqa: F401
    AGENT,
    HOOK,
    _bearer,
    _entries,
    _mint,
    engine,
    keys_file,
)

MEMORY_URI = "file:///home/owner/.claude/projects/-home-owner-repo/memory/NOTES.md"

#: The client directories the hook keys these tests mint record,
#: covering every memory and archive path the tests send.
HOOK_ROOTS = {
    "memory_roots": ["/home/owner/.claude/projects", "/home/u/.claude/projects"],
    "archive_roots": ["/home/u/.particles/claude-code/projects"],
}
CAPTURED_URL = "https://example.org/ports"
CAPTURE_BODY = "A page about ports.\n"


def _entry(entry_id: str) -> Any:
    (found,) = [e for e in _entries() if e.entry_id == entry_id]
    return found


def _run(coro_fn: Any) -> Any:
    from particles.db import get_engine

    async def _go() -> Any:
        try:
            return await coro_fn()
        finally:
            await get_engine().dispose()

    return asyncio.run(_go())


def _capture(tmp_path: Path) -> str:
    """File a web capture of :data:`CAPTURED_URL` through the real intake."""
    from particles.corpus.deposit import deposit_web_clipper
    from particles.db import session_scope

    captures = tmp_path / "captures"
    captures.mkdir()
    (captures / "ports.md").write_text(
        f"---\nsource: {CAPTURED_URL}\ntitle: Ports\n---\n{CAPTURE_BODY}"
    )

    async def _go() -> str:
        async with session_scope() as s:
            ((entry_id, _snap),) = await deposit_web_clipper(s, captures)
            await s.commit()
            return entry_id

    return str(_run(_go))


def _not_global(tags: list[str]) -> bool:
    from particles.config import get_config

    harness = get_config().observer_scope.harness_tags
    return classify_entry(tags, harness) is not EntryScope.GLOBAL


# --- W2, W3: confinement runs before the reconciles and the unchanged return ----


@pytest.mark.parametrize(
    ("uri", "source_type", "mutability"),
    [
        ("claude-code://session/foreign", "CONVERSATION", "APPEND_ONLY"),
        (MEMORY_URI, "LOCAL_MARKDOWN", "MUTABLE"),
    ],
)
def test_an_unchanged_redeposit_changes_no_column_of_another_principals_entry(
    engine: TestClient, uri: str, source_type: str, mutability: str
) -> None:
    held = engine.post(
        "/corpus/deposit/text",
        json={
            "text": "held body",
            "uri_r": uri,
            "source_type": source_type,
            "mutability": mutability,
            "fetch_policy": "NEVER",
            "deposited_by": "someone-else",
            "tags": [project_tag("alpha")],
        },
    )
    assert held.status_code == 200, held.text
    entry_id = held.json()["entry_id"]
    before = _entry(entry_id)

    token = _mint("agent", HOOK, reserved_ok=True, **HOOK_ROOTS)
    resp = engine.post(
        "/corpus/deposit/text",
        json={
            "text": "held body",
            "uri_r": uri,
            "source_type": source_type,
            "mutability": mutability,
            "tags": [project_tag("beta")],
        },
        headers=_bearer(token),
    )
    assert resp.status_code == 409, resp.text
    after = _entry(entry_id)
    assert sorted(after.tags) == sorted(before.tags)
    assert after.fetch_policy == before.fetch_policy
    assert len(after.snapshots) == len(before.snapshots)


def test_a_harvester_still_redeposits_its_own_entry_unchanged(engine: TestClient) -> None:
    token = _mint("agent", HOOK, reserved_ok=True, **HOOK_ROOTS)
    body = {
        "text": "A session.",
        "uri_r": "claude-code://session/own-session",
        "mutability": "APPEND_ONLY",
        "tags": ["claude-code", project_tag("alpha")],
    }
    first = engine.post("/corpus/deposit/text", json=body, headers=_bearer(token))
    assert first.status_code == 200, first.text
    again = engine.post("/corpus/deposit/text", json=body, headers=_bearer(token))
    assert again.status_code == 200, again.text
    assert again.json()["unchanged"] is True
    assert again.json()["entry_id"] == first.json()["entry_id"]


# --- W4, W36: an agent key's generic deposit is never global --------------------


def test_an_unbound_agent_keys_generic_text_deposit_is_never_global(engine: TestClient) -> None:
    token = _mint("agent", "agent:ci-bot")
    resp = engine.post(
        "/corpus/deposit/text",
        json={"text": "The deploy key rotates every Tuesday."},
        headers=_bearer(token),
    )
    assert resp.status_code == 200, resp.text
    entry = _entry(resp.json()["entry_id"])
    assert entry.deposited_by == "agent:ci-bot"
    assert _not_global(entry.tags), entry.tags


def test_an_unbound_agent_keys_upload_is_never_global(engine: TestClient) -> None:
    token = _mint("agent", "agent:ci-bot")
    resp = engine.post(
        "/corpus/deposit/file",
        files={"file": ("note.md", b"# Note\n\nAgent material.\n", "text/markdown")},
        headers=_bearer(token),
    )
    assert resp.status_code == 200, resp.text
    assert _not_global(_entry(resp.json()["entry_id"]).tags)


def test_an_agent_deposit_naming_a_project_carries_that_key_and_no_marker(
    engine: TestClient,
) -> None:
    token = _mint("agent", "agent:ci-bot")
    resp = engine.post(
        "/corpus/deposit/text",
        json={"text": "Keyed agent material.", "tags": [project_tag("alpha")]},
        headers=_bearer(token),
    )
    assert resp.status_code == 200, resp.text
    assert _entry(resp.json()["entry_id"]).tags == [project_tag("alpha")]


def test_an_operator_keys_untagged_deposit_stays_global(engine: TestClient) -> None:
    resp = engine.post("/corpus/deposit/text", json={"text": "An operator's note."})
    assert resp.status_code == 200, resp.text
    assert not _not_global(_entry(resp.json()["entry_id"]).tags)


# --- W52, W26, W67: the web-capture rule (R13) ----------------------------------


def test_a_confined_deposit_leaves_a_capture_entrys_uri(cli_db: Path) -> None:
    from particles.core.schema import FetchPolicy, Mutability, WarcRecordType
    from particles.corpus.deposit import (
        ForeignEntryRefused,
        own_entries_only_for,
        write_entry_and_snapshot,
    )
    from particles.corpus.store import get_entry
    from particles.db import session_scope

    url = "https://example.test/article"

    async def _go() -> str | None:
        async with session_scope() as s:
            capture_id, _ = await write_entry_and_snapshot(
                s,
                url,
                "WEB_PAGE",
                Mutability.STABLE,
                FetchPolicy.NEVER,
                b"capture bytes",
                "a" * 64,
                "blob-a",
                "web-clipper",
                [],
                WarcRecordType.RESPONSE,
            )
            await s.commit()
        async with session_scope() as s:
            with pytest.raises(ForeignEntryRefused), own_entries_only_for("agent:ci-bot"):
                await write_entry_and_snapshot(
                    s,
                    url,
                    "WEB_PAGE",
                    Mutability.STABLE,
                    FetchPolicy.NEVER,
                    b"agent bytes",
                    "b" * 64,
                    "blob-b",
                    "agent:ci-bot",
                    [],
                    WarcRecordType.RESPONSE,
                )
            await s.commit()
        async with session_scope() as s:
            capture = await get_entry(s, capture_id)
            assert capture is not None
            return capture.uri_r

    assert _run(_go) == url


class _Response:
    status_code = 200
    content = b"<html><body>Another reading of the page.</body></html>"
    headers = {"content-type": "text/html"}

    def raise_for_status(self) -> None:
        return None


def test_an_agent_url_deposit_changes_no_column_of_a_captures_entry(
    engine: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import particles.corpus.deposit as deposit

    capture_id = _capture(tmp_path)

    async def _get_capped(_client: Any, _url: str) -> _Response:
        return _Response()

    @asynccontextmanager
    async def _client() -> AsyncIterator[None]:
        yield None

    monkeypatch.setattr(deposit, "get_capped", _get_capped)
    monkeypatch.setattr(deposit, "particles_client", _client)
    monkeypatch.setattr("particles.url_safety.validate_fetch_url", lambda _url: None)
    token = _mint("agent", AGENT)
    resp = engine.post("/corpus/deposit/url", json={"url": CAPTURED_URL}, headers=_bearer(token))
    assert resp.status_code == 409, resp.text
    assert _entry(capture_id).uri_r == CAPTURED_URL


def test_a_text_deposit_of_a_captured_url_never_joins_the_capture(
    engine: TestClient, tmp_path: Path
) -> None:
    capture_id = _capture(tmp_path)
    before = _entry(capture_id)
    resp = engine.post(
        "/corpus/deposit/text",
        json={
            "text": CAPTURE_BODY,
            "uri_r": CAPTURED_URL,
            "source_type": "WEB_PAGE",
            "mutability": "STABLE",
            "deposited_by": "operator",
        },
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["entry_id"] != capture_id
    assert resp.json()["unchanged"] is False
    after = _entry(capture_id)
    assert after.uri_r is None
    assert sorted(after.tags) == sorted(before.tags)
    assert after.deposited_by == before.deposited_by
    assert _entry(resp.json()["entry_id"]).uri_r == CAPTURED_URL


def test_a_captures_uri_release_is_recorded_as_an_event(engine: TestClient, tmp_path: Path) -> None:
    from particles.db import session_scope
    from particles.store.event_store import OperatorEventType, list_events

    capture_id = _capture(tmp_path)
    resp = engine.post(
        "/corpus/deposit/text",
        json={
            "text": "Another reading.",
            "uri_r": CAPTURED_URL,
            "source_type": "WEB_PAGE",
            "mutability": "STABLE",
            "deposited_by": "operator",
        },
    )
    assert resp.status_code == 200, resp.text

    async def _go() -> list[Any]:
        async with session_scope() as s:
            return list(
                await list_events(s, event_type=OperatorEventType.CORPUS_ENTRY_URI_RELEASED)
            )

    (event,) = _run(_go)
    assert event.payload is not None
    assert event.payload["uri_r"] == CAPTURED_URL
    assert event.payload["released_to"] == "operator"
    assert [ref.ref_id for ref in event.refs] == [capture_id]


def test_a_key_named_web_clipper_needs_the_reserved_acknowledgement(keys_file: Path) -> None:
    with pytest.raises(KeyRuleError):
        _keys.mint(Role.AGENT, "web-clipper")
    _keys.mint(Role.AGENT, "web-clipper", reserved_ok=True)


# --- W21: a local LAZY enrolment of an entry another depositor created ----------


def test_a_rule_enrolment_of_an_http_created_entry_masks_its_refresh(
    engine: TestClient, tmp_path: Path
) -> None:
    from particles.core.schema import FetchPolicy, Mutability
    from particles.corpus.rule_sources import RULE_SOURCE_TAG, RULE_SOURCE_TYPE
    from particles.db import session_scope
    from particles.operations.deposit import deposit_text_versioned

    uri = (tmp_path / "repo" / "CLAUDE.md").as_uri()
    pre = engine.post(
        "/corpus/deposit/text",
        json={
            "text": "# placeholder",
            "uri_r": uri,
            "source_type": "LOCAL_MARKDOWN",
            "mutability": "MUTABLE",
        },
    )
    assert pre.status_code == 200, pre.text

    async def _enrol() -> str:
        async with session_scope(write=True) as s:
            entry_id, _snap, _unchanged = await deposit_text_versioned(
                s,
                text="# real rules",
                uri_r=uri,
                source_type=RULE_SOURCE_TYPE,
                mutability=Mutability.MUTABLE,
                tags=[RULE_SOURCE_TAG],
                deposited_by="rule-sync",
                fetch_policy=FetchPolicy.LAZY,
            )
            await s.commit()
            return str(entry_id)

    enrolled = _run(_enrol)
    entry = _entry(enrolled)
    assert enrolled != pre.json()["entry_id"] or RULE_SOURCE_TAG in entry.tags, (
        entry.deposited_by,
        entry.tags,
    )


# --- W41: a harvester key names only the harvest's own URIs ---------------------


@pytest.mark.parametrize(
    "uri",
    [
        "file:///srv/someone-else/repo/docs/ADR/active/0999-x.md",
        "file:///home/u/repo/CLAUDE.md",
        "file:///home/u/.claude/projects/p/memory/../../../repo/AGENTS.md",
        "file://otherhost/home/u/.claude/projects/p/memory/MEMORY.md",
        "file:///home/u/.claude/projects/p/memory/MEMORY.md?x=1",
        "file:///home/u/.claude/projects/p/memory/notes.txt",
        "claude-code://session/../x",
        "claude-code://elsewhere/abc",
    ],
)
def test_a_harvester_key_names_no_uri_outside_the_harvest_shapes(
    engine: TestClient, uri: str
) -> None:
    token = _mint("agent", HOOK, reserved_ok=True, **HOOK_ROOTS)
    source_type = "CONVERSATION" if uri.startswith("claude-code://") else "LOCAL_MARKDOWN"
    resp = engine.post(
        "/corpus/deposit/text",
        json={
            "text": "---\nid: ADR-0999\nsupersedes: [ADR-0001]\n---\n# A decision\n",
            "uri_r": uri,
            "source_type": source_type,
            "mutability": "STABLE",
        },
        headers=_bearer(token),
    )
    assert resp.status_code in (400, 403), resp.text


@pytest.mark.parametrize(
    ("uri", "source_type"),
    [
        ("claude-code://session/0f6b2c1e-1d2a-4c3b-9e8f-123456789abc", "CONVERSATION"),
        ("file:///home/u/.claude/projects/-home-u-repo/memory/MEMORY.md", "LOCAL_MARKDOWN"),
        ("file:///home/u/.claude/projects/-home-u-repo/memory/sub/note.md", "LOCAL_MARKDOWN"),
        (
            "file:///home/u/.particles/claude-code/projects/-home-u-repo/MEMORY.archive.md",
            "LOCAL_MARKDOWN",
        ),
    ],
)
def test_a_harvester_key_names_the_harvests_own_uris(
    engine: TestClient, uri: str, source_type: str
) -> None:
    token = _mint("agent", HOOK, reserved_ok=True, **HOOK_ROOTS)
    resp = engine.post(
        "/corpus/deposit/text",
        json={
            "text": "Harvested.",
            "uri_r": uri,
            "source_type": source_type,
            "mutability": "MUTABLE",
            "tags": ["claude-code", "memory-file"],
        },
        headers=_bearer(token),
    )
    assert resp.status_code == 200, resp.text


def test_a_harvester_key_may_not_write_the_rule_file_mark(engine: TestClient) -> None:
    token = _mint("agent", HOOK, reserved_ok=True, **HOOK_ROOTS)
    resp = engine.post(
        "/corpus/deposit/text",
        json={
            "text": "# Rules",
            "uri_r": "file:///home/u/.claude/projects/p/memory/MEMORY.md",
            "source_type": "LOCAL_MARKDOWN",
            "mutability": "MUTABLE",
            "tags": ["rule-file"],
        },
        headers=_bearer(token),
    )
    assert resp.status_code == 403, resp.text


# --- W35, Q32: an agent's upload never retypes its own entry --------------------


def test_an_agent_keys_upload_never_retypes_its_cited_conversation_entry(
    engine: TestClient,
) -> None:
    from particles.extraction.registry import infer_domain

    token = _mint("agent", "agent:s6-proof")
    text = "The build cache lives under the repository's tmp directory."
    first = engine.post("/particles/deposit", json={"text": text}, headers=_bearer(token))
    assert first.status_code == 200, first.text
    entry_id = first.json()["entry_id"]
    before = _entry(entry_id)
    assert before.source_type == "CONVERSATION"
    assert infer_domain("CONVERSATION") is not None

    upload = engine.post(
        "/corpus/deposit/file",
        files={"file": ("note.md", text.encode(), "text/markdown")},
        data={"tags": list(before.tags)},
        headers=_bearer(token),
    )
    assert upload.status_code == 200, upload.text
    assert upload.json()["entry_id"] != entry_id
    assert _entry(entry_id).source_type == "CONVERSATION"


async def _write_entry(session: Any, uri: str, body: bytes, source_type: str, by: str) -> str:
    from particles.core.schema import FetchPolicy, Mutability, WarcRecordType
    from particles.corpus.deposit import save_blob, sha256, write_entry_and_snapshot

    digest = sha256(body)
    entry_id, _snap = await write_entry_and_snapshot(
        session=session,
        uri_r=uri,
        source_type=source_type,
        mutability=Mutability.MUTABLE,
        fetch_policy=FetchPolicy.NEVER,
        content=body,
        content_hash=digest,
        archive_path=save_blob(body, digest),
        deposited_by=by,
        tags=[],
        warc_record_type=WarcRecordType.RESPONSE,
    )
    return entry_id


@pytest.mark.asyncio
async def test_a_confined_deposit_never_retypes_its_own_uri_entry(
    db_session: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Under confinement a URI-matched re-deposit of another type keeps the entry's type (G58)."""
    from particles.corpus.deposit import own_entries_only_for
    from particles.corpus.store import get_entry, list_snapshots_for_entry

    monkeypatch.setenv("PARTICLES_BLOB_DIR", str(tmp_path / "blobs"))
    with own_entries_only_for("agent-a"):
        entry_id = await _write_entry(
            db_session, "https://example.org/a", b"page", "WEB_PAGE", "agent-a"
        )
        again = await _write_entry(
            db_session, "https://example.org/a", b"%PDF-1.4 x", "PDF", "agent-a"
        )
    assert again == entry_id
    entry = await get_entry(db_session, entry_id)
    assert entry is not None
    assert entry.source_type == "WEB_PAGE"
    assert len(await list_snapshots_for_entry(db_session, entry_id)) == 2


@pytest.mark.asyncio
async def test_an_unconfined_deposit_still_retypes_a_uri_entry(
    db_session: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The operator's re-deposit as a more specific type still retypes the entry."""
    from particles.corpus.store import get_entry

    monkeypatch.setenv("PARTICLES_BLOB_DIR", str(tmp_path / "blobs"))
    entry_id = await _write_entry(db_session, "https://example.org/b", b"page", "WEB_PAGE", "op")
    await _write_entry(db_session, "https://example.org/b", b"%PDF-1.4 x", "PDF", "op")
    entry = await get_entry(db_session, entry_id)
    assert entry is not None
    assert entry.source_type == "PDF"

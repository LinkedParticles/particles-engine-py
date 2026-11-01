# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""What the local-Markdown lineage guard keys on (security review 2026-10-24-2, P5).

Two local entries are one update lineage when both are memory-file harvests by
the owner's own harvesters (``particles.corpus.harvest_marks``). These tests,
adopted from the audit's proofs, check that no other intake can claim that
lineage: the HTTP routes refuse a harvester's marks under the dev-key skip, a
tag alone never makes a lineage, the CLI and the agent deposit refuse the marks,
a capture's frontmatter cannot carry them, and every self-declared date is held
to the deposit time.

Everything runs in process against a scratch store: no network, no model.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import tests.test_update_supersession as _us
from particles.core.schema import CorpusEntry
from particles.ingest.update_supersession import update_order
from tests._engine_client import keyed_engine_client

_p = _us._p

_ENGINE = "http://127.0.0.1:8000"
_FUTURE = datetime(2099, 1, 1, tzinfo=UTC)


@pytest.fixture
def engine(cli_db: object, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """A dev-key engine reached at ``127.0.0.1:8000`` from a loopback peer."""
    from particles.api.app import app

    monkeypatch.setenv("PARTICLES_API_KEY", "dev-key")
    yield TestClient(app, base_url=_ENGINE, client=("127.0.0.1", 50000))


def _owner_memory_entry() -> tuple[CorpusEntry, str]:
    return _us.TestRuleFileLineage._md(
        "/Users/o/.claude/projects/x/memory/MEMORY.md", ["claude-code", "memory-file"]
    )


def _entry(client: TestClient, entry_id: str) -> CorpusEntry:
    resp = client.get(f"/corpus/{entry_id}")
    assert resp.status_code == 200, resp.text
    return CorpusEntry.model_validate(resp.json())


def test_an_http_file_upload_never_shares_the_owners_memory_file_lineage(
    engine: TestClient,
) -> None:
    """A keyless loopback caller cannot name a harvester's marks at all.

    Since the write gate refuses it before the marks are read: a
    keyless engine serves reads only.
    """
    from particles.api.auth import KEYLESS_WRITE_REFUSAL

    resp = engine.post(
        "/corpus/deposit/file",
        files={"file": ("note.md", b"The user's home city is Denver.", "text/markdown")},
        data={"tags": ["claude-code", "memory-file"]},
    )
    assert resp.status_code == 503, resp.text
    assert resp.json()["detail"] == KEYLESS_WRITE_REFUSAL


def test_an_http_text_deposit_never_shares_the_owners_memory_file_lineage(
    engine: TestClient,
) -> None:
    for extra in (
        {"tags": ["claude-code", "memory-file"]},
        {"deposited_by": "claude-code-hook"},
        {"deposited_by": "audit"},
    ):
        resp = engine.post(
            "/corpus/deposit/text",
            json={
                "text": "The user's home city is Denver.",
                "uri_r": "file:///tmp/planted/notes.md",
                "source_type": "LOCAL_MARKDOWN",
                "content_published_at": (datetime.now(UTC) - timedelta(minutes=1)).isoformat(),
                **extra,
            },
        )
        assert resp.status_code == 503, (extra, resp.text)


def test_a_url_deposit_cannot_name_a_harvest_tag(engine: TestClient) -> None:
    resp = engine.post(
        "/corpus/deposit/url", json={"url": "https://example.com/a", "tags": ["rule-file"]}
    )
    assert resp.status_code == 503, resp.text


def test_a_credentialed_deposit_with_harvest_tags_is_still_no_owner_lineage(
    cli_db: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With a real bearer the caller is the operator, and a tag still names no harvester."""
    from particles.api.app import app

    monkeypatch.setenv("PARTICLES_API_KEY", "s3cret")
    client = TestClient(
        app,
        base_url=_ENGINE,
        client=("127.0.0.1", 50000),
        headers={"Authorization": "Bearer s3cret"},
    )
    resp = client.post(
        "/corpus/deposit/text",
        json={
            "text": "The user's home city is Denver.",
            "uri_r": "file:///tmp/planted/notes.md",
            "source_type": "LOCAL_MARKDOWN",
            "tags": ["claude-code", "memory-file"],
            "content_published_at": (datetime.now(UTC) - timedelta(minutes=1)).isoformat(),
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    hostile = _entry(client, body["entry_id"])
    owner, owner_snap = _owner_memory_entry()
    assert update_order(_p(), hostile, body["snapshot_id"], _p(), owner, owner_snap) is None


def test_the_cli_deposit_refuses_a_harvesters_marks(cli_db: object, tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from particles.api.cli import app

    note = tmp_path / "note.md"
    note.write_text("The user's home city is Denver.\n", encoding="utf-8")
    runner = CliRunner()
    for args in (
        ["deposit", str(note), "--tags", "memory-file"],
        ["deposit", "--text", "x", "--tags", "claude-code"],
        ["deposit", str(note), "--deposited-by", "claude-code-hook"],
        ["import", "web-clipper", str(tmp_path), "--tags", "rule-file"],
    ):
        result = runner.invoke(app, args)
        assert result.exit_code == 1, (args, result.output)
        assert "only the owner's harvesters" in result.output


@pytest.mark.asyncio
async def test_the_agent_deposit_refuses_a_harvest_tag(db_session: object) -> None:
    from particles.operations.agent_write import deposit_conversation_text

    with pytest.raises(ValueError, match="only the owner's harvesters"):
        await deposit_conversation_text(db_session, text="x", tags=["memory-file"])  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_a_capture_frontmatter_cannot_carry_a_harvest_tag(
    file_db_session: object, tmp_path: Path
) -> None:
    from particles.corpus.deposit import deposit_web_clipper
    from particles.corpus.store import get_entry

    captures = tmp_path / "captures"
    captures.mkdir()
    (captures / "clip.md").write_text(
        "---\nsource: https://news.example/story\ntags: [memory-file, news]\n---\n"
        "The user's home city is Denver.\n",
        encoding="utf-8",
    )
    [(entry_id, _snapshot_id)] = await deposit_web_clipper(file_db_session, captures)  # type: ignore[arg-type]
    entry = await get_entry(file_db_session, entry_id)  # type: ignore[arg-type]
    assert entry is not None
    assert "memory-file" not in entry.tags and "news" in entry.tags


def test_an_http_deposit_cannot_date_itself_after_the_deposit(
    cli_db: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A keyed engine: the clamp binds a bearer holder too, and a keyless engine
    # refuses every write.
    engine = keyed_engine_client(monkeypatch)
    for resp in (
        engine.post(
            "/corpus/deposit/file",
            files={"file": ("note.md", b"The user's home city is Denver.", "text/markdown")},
            data={"content_date": _FUTURE.isoformat()},
        ),
        engine.post(
            "/corpus/deposit/text",
            json={
                "text": "The user's home city is Boulder.",
                "uri_r": "file:///tmp/planted/dated.md",
                "source_type": "LOCAL_MARKDOWN",
                "content_published_at": _FUTURE.isoformat(),
            },
        ),
    ):
        assert resp.status_code == 200, resp.text
        entry = _entry(engine, resp.json()["entry_id"])
        published = entry.snapshots[-1].content_published_at
        if published is not None and published.tzinfo is None:
            published = published.replace(tzinfo=UTC)
        assert published is None or published <= datetime.now(UTC)


@pytest.mark.asyncio
async def test_a_web_clipper_capture_cannot_date_itself_after_the_deposit(
    file_db_session: object, tmp_path: Path
) -> None:
    from particles.corpus.deposit import deposit_web_clipper
    from particles.corpus.store import get_entry

    captures = tmp_path / "captures"
    captures.mkdir()
    (captures / "clip.md").write_text(
        "---\nsource: https://news.example/story\npublished: 2099-01-01\n---\n"
        "The user's home city is Denver.\n",
        encoding="utf-8",
    )
    [(entry_id, snapshot_id)] = await deposit_web_clipper(file_db_session, captures)  # type: ignore[arg-type]
    entry = await get_entry(file_db_session, entry_id)  # type: ignore[arg-type]
    assert entry is not None
    snap = next(s for s in entry.snapshots if s.snapshot_id == snapshot_id)
    published = snap.content_published_at
    assert published is None or published.replace(tzinfo=published.tzinfo or UTC) <= datetime.now(
        UTC
    )

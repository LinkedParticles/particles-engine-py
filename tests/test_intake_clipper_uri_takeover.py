# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0
"""A web-clipper capture's frontmatter cannot pick an engine-enrolled entry (security review P21).

Safe properties: a capture never appends to, or retypes, an entry the engine
enrolled from its own disk; and a capture's self-declared date never lies in
the future.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from particles.core.schema import FetchPolicy, Mutability
from particles.corpus.deposit import deposit_text_versioned, deposit_web_clipper
from particles.corpus.rule_sources import RULE_SOURCE_TYPE
from particles.corpus.store import get_entry, get_latest_snapshot, list_snapshots_for_entry


async def test_capture_cannot_join_an_engine_enrolled_local_entry(
    db_session: Any, tmp_path: Path
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    rule = repo / "AGENTS.md"
    rule.write_text("the owner's rule\n", encoding="utf-8")
    uri = rule.resolve().as_uri()
    entry_id, _snap, _unchanged = await deposit_text_versioned(
        db_session,
        text="the owner's rule\n",
        uri_r=uri,
        source_type=RULE_SOURCE_TYPE,
        mutability=Mutability.MUTABLE,
        deposited_by="rule-sync",
        fetch_policy=FetchPolicy.LAZY,
    )
    await db_session.commit()

    captures = tmp_path / "captures"
    captures.mkdir()
    (captures / "clip.md").write_text(
        f"---\nsource: {uri}\n---\nsomeone else's text\n", encoding="utf-8"
    )
    await deposit_web_clipper(db_session, captures)
    await db_session.commit()

    entry = await get_entry(db_session, entry_id)
    assert entry is not None
    assert entry.source_type == RULE_SOURCE_TYPE
    assert len(await list_snapshots_for_entry(db_session, entry_id)) == 1


async def test_capture_published_date_is_not_in_the_future(db_session: Any, tmp_path: Path) -> None:
    captures = tmp_path / "captures"
    captures.mkdir()
    (captures / "clip.md").write_text(
        "---\nsource: https://example.org/page\npublished: 2099-01-01\n---\nbody\n",
        encoding="utf-8",
    )
    results = await deposit_web_clipper(db_session, captures)
    await db_session.commit()
    ((entry_id, _snap),) = results
    latest = await get_latest_snapshot(db_session, entry_id)
    assert latest is not None
    published = latest.content_published_at
    assert published is not None
    if published.tzinfo is None:
        published = published.replace(tzinfo=UTC)
    assert published <= datetime.now(UTC) + timedelta(minutes=1)


async def test_a_capture_with_a_rule_files_body_gets_its_own_entry(
    db_session: Any, tmp_path: Path
) -> None:
    body = "the owner's rule\n"
    repo = tmp_path / "repo"
    repo.mkdir()
    entry_id, _snap, _unchanged = await deposit_text_versioned(
        db_session,
        text=body,
        uri_r=(repo / "AGENTS.md").resolve().as_uri(),
        source_type=RULE_SOURCE_TYPE,
        mutability=Mutability.MUTABLE,
        deposited_by="rule-sync",
        fetch_policy=FetchPolicy.LAZY,
    )
    await db_session.commit()

    captures = tmp_path / "captures"
    captures.mkdir()
    (captures / "clip.md").write_text(body, encoding="utf-8")
    ((captured_id, _s),) = await deposit_web_clipper(db_session, captures)
    await db_session.commit()

    assert captured_id != entry_id
    entry = await get_entry(db_session, entry_id)
    assert entry is not None and entry.source_type == RULE_SOURCE_TYPE


async def _operator_web_entry(session: Any, url: str, versions: list[str]) -> str:
    entry_id = ""
    for text in versions:
        entry_id, _snap, _unchanged = await deposit_text_versioned(
            session,
            text=text,
            uri_r=url,
            source_type="WEB_PAGE",
            mutability=Mutability.STABLE,
            deposited_by="operator",
            fetch_policy=FetchPolicy.LAZY,
        )
    await session.commit()
    return entry_id


async def test_a_capture_cannot_append_to_another_depositors_web_entry(
    db_session: Any, tmp_path: Path
) -> None:
    """A capture naming another depositor's http(s) entry adds no snapshot to it (Q20)."""
    url = "https://example.invalid/release-notes"
    entry_id = await _operator_web_entry(db_session, url, ["version 2 removes the legacy API\n"])
    captures = tmp_path / "captures"
    captures.mkdir()
    (captures / "clip.md").write_text(
        f"---\nsource: {url}\n---\nversion 2 keeps the legacy API forever\n", encoding="utf-8"
    )

    ((captured_id, _s),) = await deposit_web_clipper(db_session, captures)
    await db_session.commit()

    assert captured_id != entry_id
    assert len(await list_snapshots_for_entry(db_session, entry_id)) == 1
    entry = await get_entry(db_session, entry_id)
    assert entry is not None and entry.source_type == "WEB_PAGE"
    captured = await get_entry(db_session, captured_id)
    assert captured is not None and (captured.uri_r or "").startswith("file://")


async def test_a_capture_cannot_revert_another_depositors_web_entry_by_its_body(
    db_session: Any, tmp_path: Path
) -> None:
    """A capture holding an older snapshot of another depositor's page is its own entry (Q20)."""
    old, new = "the API is deprecated\n", "the API is removed\n"
    entry_id = await _operator_web_entry(db_session, "https://example.invalid/notes", [old, new])
    captures = tmp_path / "captures"
    captures.mkdir()
    (captures / "clip.md").write_text(
        f"---\nsource: https://example.invalid/mirror\n---\n{old}", encoding="utf-8"
    )

    ((captured_id, _s),) = await deposit_web_clipper(db_session, captures)
    await db_session.commit()

    assert captured_id != entry_id
    assert len(await list_snapshots_for_entry(db_session, entry_id)) == 2


async def test_a_clippers_own_web_entry_still_collects_its_recaptures(
    db_session: Any, tmp_path: Path
) -> None:
    """A page clipped twice by the same depositor still collapses onto one entry."""
    captures = tmp_path / "captures"
    captures.mkdir()
    clip = captures / "clip.md"
    clip.write_text("---\nsource: https://example.org/page\n---\nfirst\n", encoding="utf-8")
    ((first_id, _s),) = await deposit_web_clipper(db_session, captures)
    await db_session.commit()
    clip.write_text("---\nsource: https://example.org/page\n---\nsecond\n", encoding="utf-8")
    ((second_id, _s2),) = await deposit_web_clipper(db_session, captures)
    await db_session.commit()

    assert second_id == first_id
    assert len(await list_snapshots_for_entry(db_session, first_id)) == 2


class _Resp:
    status_code = 200
    content = b"<html><body>the operator's fetched page</body></html>"
    headers = {"content-type": "text/html; charset=utf-8"}

    def raise_for_status(self) -> None:
        return None


class _Client:
    async def __aenter__(self) -> _Client:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None


def _stub_fetch(monkeypatch: Any) -> None:
    """Serve ``_Resp`` to ``deposit_url`` in process; no request leaves the test."""
    import particles.corpus.deposit as deposit_module
    import particles.ingest.importers.registry as registry
    import particles.url_safety as url_safety

    async def _get_capped(_client: Any, _url: str) -> _Resp:
        return _Resp()

    monkeypatch.setattr(url_safety, "validate_fetch_url", lambda _u: None)
    monkeypatch.setattr(deposit_module, "particles_client", lambda **_k: _Client())
    monkeypatch.setattr(deposit_module, "get_capped", _get_capped)
    monkeypatch.setattr(registry, "get_importers", lambda: [])


async def test_an_operator_url_deposit_never_joins_an_entry_a_capture_created(
    db_session: Any, tmp_path: Path, monkeypatch: Any
) -> None:
    """A capture cannot pre-claim a URL a later deposit then lands on (R13)."""
    from particles.corpus.deposit import deposit_url

    url = "https://example.org/article"
    captures = tmp_path / "captures"
    captures.mkdir()
    (captures / "clip.md").write_text(
        f"---\nsource: {url}\ntags: [project:someone-else]\n---\nwhat the capture says\n",
        encoding="utf-8",
    )
    ((clip_entry, _),) = await deposit_web_clipper(db_session, captures)
    await db_session.commit()
    _stub_fetch(monkeypatch)

    entry_id, _snap = await deposit_url(db_session, url, deposited_by="operator")
    await db_session.commit()

    assert entry_id != clip_entry
    entry = await get_entry(db_session, entry_id)
    assert entry is not None
    assert entry.uri_r == url
    assert entry.deposited_by == "operator"
    assert "project:someone-else" not in (entry.tags or [])
    capture = await get_entry(db_session, clip_entry)
    assert capture is not None
    assert capture.uri_r is None
    assert capture.deposited_by == "web-clipper"
    assert len(await list_snapshots_for_entry(db_session, clip_entry)) == 1


async def test_a_deposit_matching_a_captures_body_gets_its_own_entry(
    db_session: Any, tmp_path: Path
) -> None:
    """The content-hash tier never joins a capture's entry for another depositor (R13)."""
    captures = tmp_path / "captures"
    captures.mkdir()
    body = "what the capture says\n"
    (captures / "clip.md").write_text(
        f"---\nsource: https://example.org/a\n---\n{body}", encoding="utf-8"
    )
    ((clip_entry, _),) = await deposit_web_clipper(db_session, captures)
    await db_session.commit()

    entry_id, _snap, _unchanged = await deposit_text_versioned(
        db_session,
        text=body,
        uri_r="https://example.org/b",
        source_type="WEB_PAGE",
        mutability=Mutability.STABLE,
        deposited_by="operator",
        fetch_policy=FetchPolicy.NEVER,
    )
    await db_session.commit()

    assert entry_id != clip_entry
    assert len(await list_snapshots_for_entry(db_session, clip_entry)) == 1

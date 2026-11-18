# SPDX-FileCopyrightText: 2026 The Particles authors
# SPDX-License-Identifier: Apache-2.0

"""A client pushes its own rule files to a remote engine (to §4).

The engine side: a scratch store and key file under ``tmp_path``, configuration
only through the loader. The client side: a fake backend that records what it
was sent, so no engine is reached. No model call, no outside connection.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from particles.corpus.harvest_marks import claude_project_slug
from tests._corpus_rows import CLIENT_HOST_ID

PROJECT = "/home/u/src/repo"
CLAUDE = "/home/u/.claude"
SLUG = claude_project_slug(PROJECT)


def _create_tables() -> None:
    async def _go() -> None:
        import particles._orm_modules  # noqa: F401
        from particles.db import Base, get_engine

        async with get_engine().begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        await get_engine().dispose()

    asyncio.run(_go())


@pytest.fixture
def engine_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Generator[Path, None, None]:
    from particles.api import _keys
    from particles.config import reset_config

    home = tmp_path / "home"
    (home / ".claude" / "projects").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    keydir = tmp_path / "keys"
    keydir.mkdir(mode=0o700)
    cfg = tmp_path / "config.yaml"
    cfg.write_text("network:\n  egress: none\n", encoding="utf-8")
    monkeypatch.setenv("PARTICLES_CONFIG", str(cfg))
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path / 'r.db'}")
    monkeypatch.setenv("PARTICLES_BLOB_DIR", str(tmp_path / "blobs"))
    monkeypatch.setenv("PARTICLES_API_KEYS_FILE", str(keydir / "engine-keys.json"))
    monkeypatch.delenv("PARTICLES_API_KEY", raising=False)
    reset_config()
    _keys.reset_key_cache()
    _keys.forget_key_files()
    _create_tables()
    yield home
    _keys.reset_key_cache()
    _keys.forget_key_files()
    reset_config()


def _client() -> TestClient:
    from particles.api.app import app

    return TestClient(app, base_url="http://127.0.0.1:8000", client=("127.0.0.1", 50000))


def _hook(**extra: Any) -> str:
    from particles.api import _keys
    from particles.api._keys import Role

    kwargs: dict[str, Any] = {
        "host_id": CLIENT_HOST_ID,
        "memory_roots": [f"{CLAUDE}/projects"],
        "rule_roots": [PROJECT],
        "global_rule_roots": [CLAUDE],
    }
    kwargs.update(extra)
    return _keys.mint(Role.AGENT, "claude-code-hook", reserved_ok=True, **kwargs)[0]


def _push(token: str, path: str, tags: list[str]) -> Any:
    return _client().post(
        "/corpus/deposit/text",
        json={
            "text": "# Rules\n\nRun the gates.\n",
            "uri_r": f"file://{path}",
            "source_type": "LOCAL_MARKDOWN",
            "mutability": "MUTABLE",
            "fetch_policy": "NEVER",
            "tags": tags,
            "deposited_by": "claude-code-hook",
        },
        headers={"Authorization": f"Bearer {token}"},
    )


def _row(entry_id: str) -> Any:
    from particles.corpus.store import CorpusEntryRow
    from particles.db import session_scope

    async def _go() -> Any:
        async with session_scope() as session:
            return await session.get(CorpusEntryRow, entry_id)

    return asyncio.run(_go())


# --- the engine admits a push (§3.4, §3.5) -------------------------------------------


def test_a_project_rule_file_under_its_root_is_accepted_with_its_tags(engine_home: Path) -> None:
    resp = _push(_hook(), f"{PROJECT}/docs/AGENTS.md", ["rule-file", f"project:{SLUG}"])
    assert resp.status_code == 200, resp.text
    row = _row(resp.json()["entry_id"])
    assert row.origin_host == CLIENT_HOST_ID
    assert "rule-file" in row.tags_json and SLUG in row.tags_json


@pytest.mark.parametrize(
    ("path", "tags", "why"),
    [
        (f"{PROJECT}/AGENTS.md", ["rule-file", "project:-other"], "only as project"),
        (f"{PROJECT}/AGENTS.md", ["rule-file"], "only as project"),
        (f"{PROJECT}/.claude/worktrees/w/AGENTS.md", ["rule-file", f"project:{SLUG}"], "excluded"),
        (f"{PROJECT}/README.md", ["rule-file", f"project:{SLUG}"], "only files named"),
        ("/home/u/src/other/AGENTS.md", ["rule-file"], "none of its rule roots"),
        (f"{CLAUDE}/CLAUDE.md", ["rule-file", f"project:{SLUG}"], "global rule file a project"),
        (f"{CLAUDE}/projects/-x/CLAUDE.md", ["rule-file"], "none of its rule roots"),
    ],
    ids=["wrong-project", "no-project", "worktree", "not-a-rule-name", "outside", "global-tag",
         "nested-global"],
)  # fmt: skip
def test_a_rule_file_push_that_breaks_the_rules_is_refused(
    engine_home: Path, path: str, tags: list[str], why: str
) -> None:
    resp = _push(_hook(), path, tags)
    assert resp.status_code == 403, resp.text
    assert why in resp.text


def test_a_global_rule_file_is_stored_global_not_unattributed(engine_home: Path) -> None:
    resp = _push(_hook(), f"{CLAUDE}/CLAUDE.md", ["rule-file"])
    assert resp.status_code == 200, resp.text
    tags = _row(resp.json()["entry_id"]).tags_json
    assert "scope:unattributed" not in tags and "project:" not in tags


def test_only_a_harvester_with_a_host_pushes_the_rule_file_mark(engine_home: Path) -> None:
    from particles.api import _keys
    from particles.api._keys import Role

    other = _keys.mint(Role.AGENT, "agent:ci-bot")[0]
    assert _push(other, f"{PROJECT}/AGENTS.md", ["rule-file"]).status_code == 403
    no_host = _hook(host_id=None)
    resp = _push(no_host, f"{PROJECT}/AGENTS.md", ["rule-file", f"project:{SLUG}"])
    assert resp.status_code == 403 and "--host-id" in resp.text


def test_a_push_continues_the_entry_rules_sync_made(engine_home: Path) -> None:
    """The 39 existing rule-file entries were deposited by rule-sync and init (§3.5)."""
    from particles.core.schema import FetchPolicy, Mutability
    from particles.corpus.deposit import deposit_text_versioned
    from particles.corpus.origin import deposit_origin
    from particles.db import session_scope

    uri = f"file://{PROJECT}/AGENTS.md"

    async def _sync() -> str:
        async with session_scope(write=True) as session:
            with deposit_origin(CLIENT_HOST_ID):
                entry_id, _, _ = await deposit_text_versioned(
                    session,
                    text="old rules",
                    uri_r=uri,
                    source_type="LOCAL_MARKDOWN",
                    mutability=Mutability.MUTABLE,
                    tags=["rule-file", f"project:{SLUG}"],
                    deposited_by="rule-sync",
                    fetch_policy=FetchPolicy.LAZY,
                )
            await session.commit()
            return entry_id

    entry_id = asyncio.run(_sync())
    resp = _push(_hook(), f"{PROJECT}/AGENTS.md", ["rule-file", f"project:{SLUG}"])
    assert resp.status_code == 200, resp.text
    assert resp.json()["entry_id"] == entry_id
    row = _row(entry_id)
    assert row.deposited_by == "rule-sync" and row.fetch_policy == "NEVER"


# --- the key's rule roots (§3.1, owner ruling 2) -------------------------------------


def test_a_rule_root_holding_another_is_refused_at_mint(engine_home: Path) -> None:
    from particles.api._keys import KeyRuleError

    with pytest.raises(KeyRuleError, match="lies under rule root"):
        _hook(rule_roots=["/home/u/src", PROJECT])


def test_a_broad_rule_root_is_refused_at_mint(engine_home: Path) -> None:
    from particles.api._keys import KeyRuleError

    with pytest.raises(KeyRuleError, match="too broad"):
        _hook(rule_roots=[str(engine_home)], global_rule_roots=None)


def test_whoami_returns_the_rule_roots(engine_home: Path) -> None:
    body = _client().get("/auth/whoami", headers={"Authorization": f"Bearer {_hook()}"}).json()
    assert body["rule_roots"] == [PROJECT] and body["global_rule_roots"] == [CLAUDE]


# --- the client's push (§4) ----------------------------------------------------------


class _FakeBackend:
    remote = True

    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []

    async def deposit_text_at_uri(self, **kwargs: Any) -> Any:
        self.sent.append(kwargs)
        return None


@pytest.fixture
def client_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Generator[Path, None, None]:
    from particles.config import reset_config

    home = tmp_path / "client"
    (home / ".claude").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    cfg = tmp_path / "client.yaml"
    cfg.write_text(
        "engine:\n  base_url: http://engine.test:8000\n"
        f"claude_code:\n  state_dir: {home / '.particles' / 'claude-code'}\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("PARTICLES_CONFIG", str(cfg))
    reset_config()
    yield home
    reset_config()


def _repo(root: Path) -> Path:
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    (root / "AGENTS.md").write_text("# Main rules\n")
    subprocess.run(["git", "-C", str(root), "add", "AGENTS.md"], check=True)
    subprocess.run(
        ["git", "-C", str(root), "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "x"],
        check=True,
    )
    return root


def _fake(monkeypatch: pytest.MonkeyPatch) -> _FakeBackend:
    import particles.api.client as client

    fake = _FakeBackend()
    monkeypatch.setattr(client, "get_backend", lambda: fake)
    return fake


def _push_files(root: Path) -> Any:
    from particles.api.cli._rule_push import push_rule_files, session_rule_files

    files, roots = session_rule_files(root)
    return asyncio.run(
        push_rule_files(files, store="default", roots=roots, filter_text=lambda text: text)
    )


def test_a_quiet_run_sends_nothing_and_a_changed_file_is_sent_once(
    client_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _fake(monkeypatch)
    root = _repo(client_home / "src" / "repo")
    (client_home / ".claude" / "CLAUDE.md").write_text("# Mine\n")
    first = _push_files(root)
    assert len(first.pushed) == 2 and len(fake.sent) == 2
    assert {call["fetch_policy"] for call in fake.sent} == {"NEVER"}
    assert _push_files(root).pushed == [] and len(fake.sent) == 2
    agents = root / "AGENTS.md"
    os.utime(agents, ns=(agents.stat().st_atime_ns, agents.stat().st_mtime_ns + 10**9))
    assert _push_files(root).pushed == [] and len(fake.sent) == 2  # same bytes
    agents.write_text("# Main rules, edited\n")
    assert [p.name for p in _push_files(root).pushed] == ["AGENTS.md"]
    assert len(fake.sent) == 3


def test_another_engine_is_sent_everything_once(
    client_home: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from particles.config import reset_config

    fake = _fake(monkeypatch)
    root = _repo(client_home / "src" / "repo")
    _push_files(root)
    (tmp_path / "client.yaml").write_text(
        "engine:\n  base_url: http://other.test:8000\n"
        f"claude_code:\n  state_dir: {client_home / '.particles' / 'claude-code'}\n",
        encoding="utf-8",
    )
    reset_config()
    assert len(_push_files(root).pushed) == 1 and len(fake.sent) == 2


def test_a_worktree_pushes_its_main_checkouts_files(
    client_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from particles.api.cli._claude_code import repository_root

    fake = _fake(monkeypatch)
    main = _repo(client_home / "src" / "repo")
    worktree = main / ".claude" / "worktrees" / "w"
    subprocess.run(["git", "-C", str(main), "worktree", "add", "-q", str(worktree)], check=True)
    (worktree / "AGENTS.md").write_text("# A branch's draft\n")
    _push_files(repository_root(worktree))
    assert [call["uri_r"] for call in fake.sent] == [(main / "AGENTS.md").resolve().as_uri()]
    assert fake.sent[0]["tags"] == ["rule-file", f"project:{claude_project_slug(main)}"]


def test_a_nested_claude_rule_file_is_never_pushed(
    client_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from particles.api.cli._rule_push import push_rule_files

    fake = _fake(monkeypatch)
    nested = client_home / ".claude" / "plugins" / "p" / "CLAUDE.md"
    nested.parent.mkdir(parents=True)
    nested.write_text("# A plugin's\n")
    report = asyncio.run(
        push_rule_files(
            [nested.resolve()],
            store="default",
            roots=[(client_home / ".claude").resolve()],
            filter_text=lambda text: text,
        )
    )
    assert report.skipped_nested == [nested.resolve()] and fake.sent == []


def test_a_remote_rules_sync_refuses_the_restamp_and_dry_runs_without_sending(
    client_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from particles.api.cli import app

    fake = _fake(monkeypatch)
    root = _repo(client_home / "src" / "repo")
    monkeypatch.chdir(root)
    runner = CliRunner()
    restamp = runner.invoke(app, ["rules", "sync", "--restamp-only"])
    assert restamp.exit_code == 1
    dry = runner.invoke(app, ["rules", "sync", "--dry-run"])
    assert dry.exit_code == 0, dry.output
    assert "would push" in dry.output and fake.sent == []
    real = runner.invoke(app, ["rules", "sync"])
    assert real.exit_code == 0, real.output
    assert "1 pushed" in real.output and len(fake.sent) == 1


def test_the_session_end_step_pushes_and_never_raises(
    client_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from particles.api.cli import hook

    fake = _fake(monkeypatch)
    root = _repo(client_home / "src" / "repo")
    first = asyncio.run(hook._push_session_rules("default", root))
    assert first == {"pushed": 1} and len(fake.sent) == 1
    assert asyncio.run(hook._push_session_rules("default", root)) == {"unchanged": 1}

    async def _broken(**_kwargs: Any) -> Any:
        raise RuntimeError("engine fell over")

    (root / "AGENTS.md").write_text("# Edited\n")
    monkeypatch.setattr(fake, "deposit_text_at_uri", _broken)
    assert "error" in asyncio.run(hook._push_session_rules("default", root))


def test_a_refused_file_is_not_sent_again_until_it_changes(
    client_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from particles.api.client.http import EngineHttpError

    fake = _fake(monkeypatch)
    root = _repo(client_home / "src" / "repo")

    async def _refuse(**kwargs: Any) -> Any:
        fake.sent.append(kwargs)
        raise EngineHttpError("403 outside every rule root")

    monkeypatch.setattr(fake, "deposit_text_at_uri", _refuse)
    assert len(_push_files(root).refused) == 1
    assert _push_files(root).refused == [] and len(fake.sent) == 1
    (root / "AGENTS.md").write_text("# Edited\n")
    assert len(_push_files(root).refused) == 1 and len(fake.sent) == 2


def _status_error(status: int, message: str) -> Exception:
    from particles.api.client.http import EngineHttpError

    error = EngineHttpError(f"{status} from /corpus/deposit/text: {message}")
    error.status_code = status
    return error


def test_a_rate_limited_push_defers_the_rest_and_records_nothing(
    client_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A 429 is "not now": every file waits for the next run, none is held as refused.

    The 2026-11-07 cutover rehearsal pushed 38 rule files under the engine's
    60-a-minute limit; the 37 answered 429 were recorded as refused by the key
    and never sent again.
    """
    fake = _fake(monkeypatch)
    root = _repo(client_home / "src" / "repo")
    (client_home / ".claude" / "CLAUDE.md").write_text("# Mine\n")

    async def _limited(**kwargs: Any) -> Any:
        fake.sent.append(kwargs)
        raise _status_error(429, "Rate limit exceeded for this client.")

    monkeypatch.setattr(fake, "deposit_text_at_uri", _limited)
    report = _push_files(root)
    assert report.refused == [] and report.pushed == []
    assert report.deferred == 2 and len(fake.sent) == 1

    monkeypatch.delattr(fake, "deposit_text_at_uri")  # back to the class's method
    assert len(_push_files(root).pushed) == 2


def test_an_engine_failure_is_retried_at_the_next_run(
    client_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _fake(monkeypatch)
    root = _repo(client_home / "src" / "repo")

    async def _failing(**kwargs: Any) -> Any:
        fake.sent.append(kwargs)
        raise _status_error(500, "Internal Server Error")

    monkeypatch.setattr(fake, "deposit_text_at_uri", _failing)
    first = _push_files(root)
    assert first.refused == [] and len(first.failed) == 1
    assert len(_push_files(root).failed) == 1 and len(fake.sent) == 2


def test_an_http_error_carries_its_status_code() -> None:
    import httpx

    from particles.api.client.http import EngineRefusedError, _http_error

    request = httpx.Request("POST", "http://engine.test/corpus/deposit/text")
    limited = _http_error(httpx.Response(429, json={"detail": "slow"}, request=request), "/x")
    refused = _http_error(httpx.Response(503, json={"detail": "keyless"}, request=request), "/x")
    assert limited.status_code == 429
    assert isinstance(refused, EngineRefusedError) and refused.status_code == 503


def test_a_session_outside_any_repository_pushes_only_the_global_files(
    client_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from particles.api.cli import hook

    fake = _fake(monkeypatch)
    loose = client_home / "Downloads" / "vendor"
    loose.mkdir(parents=True)
    (loose / "CLAUDE.md").write_text("# Someone else's\n")
    (client_home / ".claude" / "CLAUDE.md").write_text("# Mine\n")
    assert asyncio.run(hook._push_session_rules("default", client_home / "Downloads")) == {
        "pushed": 1
    }
    assert asyncio.run(hook._push_session_rules("default", None)) == {"unchanged": 1}
    assert [call["uri_r"] for call in fake.sent] == [
        (client_home / ".claude" / "CLAUDE.md").resolve().as_uri()
    ]

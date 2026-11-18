# SPDX-FileCopyrightText: 2026 The Particles authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: F811

"""The operator's service credentials are spent only on targets the operator chose.

Security review 2026-11-17-1, RA1: an agent key's ``POST /corpus/deposit/url``
for a github.com URL was fetched with the operator's ``GITHUB_API_KEY``, so a
confined principal could read a private repository it named through the
operator's authority, and a link a deposited post named was followed the same
way. The fix is one chokepoint, :func:`particles.http.service_credential`, which
returns nothing inside :func:`particles.http.withhold_service_credentials`; the
Engine enters that scope for a confined deposit, an agent deposit, a followed
link and the refresh of an entry one of those made.

Two layers, as in ``tests/test_egress_chokepoint.py``:

1. **Structural**: an AST walk of ``particles/`` fails when a service-secret
   getter (``get_github_api_key_optional``, ``get_numista_api_key``,
   ``get_notion_api_key``) is named anywhere but the helper, or when an
   ``os.environ`` / ``os.getenv`` read names one of their variables outside
   ``particles/secrets.py``.
2. **Behavioural**: an agent-key URL deposit, a followed link and the refresh
   of an agent's entry send no request carrying an operator credential, while
   an operator-key deposit still does.
"""

from __future__ import annotations

import ast
import base64
import ipaddress
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

import particles
from particles.http import (
    service_credential,
    service_credentials_withheld,
    withhold_service_credentials,
)
from tests import _source_tree
from tests.test_engine_keys import (  # noqa: F401
    AGENT,
    _bearer,
    _mint,
    engine,
    keys_file,
)

OPERATOR_GITHUB_KEY = "ghp_operatorPrivateRepoToken0123456789"
OPERATOR_NUMISTA_KEY = "numista-operator-key-0123456789"
PRIVATE_URL = "https://github.com/owner/private-repo/blob/main/secret.md"
GIST_URL = "https://gist.github.com/owner/0123456789abcdef0123"
NUMISTA_URL = "https://en.numista.com/catalogue/pieces12345.html"

#: The service-secret getters only the helper may call.
_SERVICE_GETTERS = frozenset(
    {"get_github_api_key_optional", "get_numista_api_key", "get_notion_api_key"}
)
#: Their environment variables, read only by ``particles/secrets.py``.
_SERVICE_VARS = frozenset({"GITHUB_API_KEY", "NUMISTA_API_KEY", "NOTION_API_KEY"})
_HELPER_MODULE = "http.py"
_HELPER_FUNCTION = "service_credential"


# ---------------------------------------------------------------------------
# The helper
# ---------------------------------------------------------------------------


class TestServiceCredential:
    def test_returns_the_secret_outside_a_withholding_scope(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("GITHUB_API_KEY", OPERATOR_GITHUB_KEY)
        monkeypatch.setenv("NUMISTA_API_KEY", OPERATOR_NUMISTA_KEY)
        assert service_credential("github") == OPERATOR_GITHUB_KEY
        assert service_credential("numista") == OPERATOR_NUMISTA_KEY
        assert service_credentials_withheld() is None

    def test_returns_none_inside_and_restores_after(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GITHUB_API_KEY", OPERATOR_GITHUB_KEY)
        monkeypatch.setenv("NUMISTA_API_KEY", OPERATOR_NUMISTA_KEY)
        monkeypatch.setenv("NOTION_API_KEY", "secret_notion")
        with withhold_service_credentials("an agent's deposit"):
            assert service_credential("github") is None
            assert service_credential("numista") is None
            assert service_credential("notion") is None
            assert service_credentials_withheld() == "an agent's deposit"
        assert service_credential("github") == OPERATOR_GITHUB_KEY

    def test_nested_scopes_keep_the_outer_reason(self) -> None:
        with withhold_service_credentials("outer"):
            with withhold_service_credentials("inner"):
                assert service_credentials_withheld() == "outer"
            assert service_credentials_withheld() == "outer"
        assert service_credentials_withheld() is None

    def test_keeps_each_secrets_missing_contract(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("GITHUB_API_KEY", raising=False)
        monkeypatch.delenv("NUMISTA_API_KEY", raising=False)
        assert service_credential("github") is None
        with pytest.raises(ValueError, match="NUMISTA_API_KEY"):
            service_credential("numista")

    def test_github_headers_carry_no_bearer_inside_the_scope(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from particles.extraction.github._shared import _api_headers

        monkeypatch.setenv("GITHUB_API_KEY", OPERATOR_GITHUB_KEY)
        assert _api_headers()["Authorization"] == f"Bearer {OPERATOR_GITHUB_KEY}"
        with withhold_service_credentials("a followed link"):
            assert "Authorization" not in _api_headers()

    def test_confinement_scopes_withhold(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from particles.corpus.deposit import own_entries_only_for
        from particles.corpus.deposit_role import agent_deposit

        monkeypatch.setenv("GITHUB_API_KEY", OPERATOR_GITHUB_KEY)
        with own_entries_only_for(AGENT):
            assert service_credential("github") is None
        with agent_deposit():
            assert service_credential("github") is None
        assert service_credential("github") == OPERATOR_GITHUB_KEY


# ---------------------------------------------------------------------------
# Structural: one reader of the service secrets
# ---------------------------------------------------------------------------


def _package_sources() -> Iterator[tuple[Path, ast.Module]]:
    root = Path(particles.__file__).parent
    for path in sorted(root.rglob("*.py")):
        yield path, _source_tree.parse(path.read_text())


def _getter_references(tree: ast.Module) -> Iterator[tuple[int, str, str | None]]:
    """``(line, name, enclosing function)`` for every reference to a service getter."""

    def walk(node: ast.AST, func: str | None) -> Iterator[tuple[int, str, str | None]]:
        for child in ast.iter_child_nodes(node):
            inner = func
            if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef):
                inner = child.name
            name: str | None = None
            if isinstance(child, ast.Name):
                name = child.id
            elif isinstance(child, ast.Attribute):
                name = child.attr
            elif isinstance(child, ast.ImportFrom):
                for alias in child.names:
                    if alias.name in _SERVICE_GETTERS:
                        yield child.lineno, alias.name, inner
            if name in _SERVICE_GETTERS:
                yield child.lineno, name, inner
            yield from walk(child, inner)

    yield from walk(tree, None)


def _env_reads_of_service_vars(tree: ast.Module) -> Iterator[int]:
    """Lines reading a service variable through ``os.environ`` / ``os.getenv``."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = ast.unparse(node.func)
            if func in {"os.environ.get", "os.getenv", "environ.get", "getenv"} and any(
                isinstance(a, ast.Constant) and a.value in _SERVICE_VARS for a in node.args
            ):
                yield node.lineno
        elif isinstance(node, ast.Subscript) and ast.unparse(node.value) in {
            "os.environ",
            "environ",
        }:
            if isinstance(node.slice, ast.Constant) and node.slice.value in _SERVICE_VARS:
                yield node.lineno


class TestOneReaderOfServiceSecrets:
    def test_getters_are_named_only_by_the_helper(self) -> None:
        offenders: list[str] = []
        root = Path(particles.__file__).parent
        for path, tree in _package_sources():
            rel = path.relative_to(root).as_posix()
            for line, name, func in _getter_references(tree):
                if rel == "secrets.py":
                    continue  # the definitions themselves
                if rel == _HELPER_MODULE and func == _HELPER_FUNCTION:
                    continue
                offenders.append(f"particles/{rel}:{line} {name} (in {func or 'module'})")
        assert not offenders, (
            "A service secret is read outside particles.http.service_credential, so it "
            "would ride a fetch whose target a confined principal or a followed link "
            "chose (security review 2026-11-17-1, RA1). Call service_credential(...) "
            "instead:\n" + "\n".join(offenders)
        )

    def test_service_variables_are_read_only_by_secrets(self) -> None:
        offenders: list[str] = []
        root = Path(particles.__file__).parent
        for path, tree in _package_sources():
            rel = path.relative_to(root).as_posix()
            if rel == "secrets.py":
                continue
            offenders += [f"particles/{rel}:{line}" for line in _env_reads_of_service_vars(tree)]
        assert not offenders, "Service secret read from the environment directly:\n" + "\n".join(
            offenders
        )

    def test_the_scan_sees_a_planted_reference(self) -> None:
        planted = ast.parse(
            "from particles.secrets import get_github_api_key_optional as g\n"
            "def f():\n    return g()\n"
            "def h():\n    import os\n    return os.environ.get('NUMISTA_API_KEY')\n"
        )
        assert [n for _, n, _ in _getter_references(planted)] == ["get_github_api_key_optional"]
        assert list(_env_reads_of_service_vars(planted)) == [6]

    def test_the_helper_still_reads_every_getter(self) -> None:
        tree = _source_tree.parse((Path(particles.__file__).parent / _HELPER_MODULE).read_text())
        seen = {name for _, name, func in _getter_references(tree) if func == _HELPER_FUNCTION}
        assert seen == _SERVICE_GETTERS


# ---------------------------------------------------------------------------
# Behavioural: the network sees no operator credential on a chosen target
# ---------------------------------------------------------------------------


def _record_github_requests(monkeypatch: pytest.MonkeyPatch) -> list[httpx.Request]:
    """Answer every request as GitHub would, through the real transport seam; return the log."""
    from particles import url_safety

    monkeypatch.setattr(
        url_safety, "_resolve_all", lambda _h: [ipaddress.ip_address("140.82.112.3")]
    )
    seen: list[httpx.Request] = []

    async def fake_handle(self: Any, request: httpx.Request) -> httpx.Response:
        seen.append(request)
        path = request.url.path
        if "/commits" in path or path.endswith("/comments"):
            return httpx.Response(200, json=[], request=request)
        if path.startswith("/gists/"):
            return httpx.Response(
                200,
                json={"id": "g", "owner": {"login": "owner"}, "files": {}},
                request=request,
            )
        if request.url.host.endswith("numista.com"):
            return httpx.Response(200, json={"id": 12345}, request=request)
        body = base64.b64encode(b"private repository contents").decode()
        return httpx.Response(
            200,
            json={"encoding": "base64", "content": body, "path": "secret.md"},
            request=request,
        )

    monkeypatch.setattr(url_safety.ValidatingTransport, "handle_async_request", fake_handle)
    return seen


def _carries_operator_credential(request: httpx.Request) -> bool:
    values = " ".join(request.headers.values())
    return OPERATOR_GITHUB_KEY in values or OPERATOR_NUMISTA_KEY in values


class TestAgentUrlDeposit:
    """The audit's failing proof, ported: an agent key's URL deposit spends no operator key."""

    @pytest.fixture(autouse=True)
    def _operator_secrets(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GITHUB_API_KEY", OPERATOR_GITHUB_KEY)
        monkeypatch.setenv("NUMISTA_API_KEY", OPERATOR_NUMISTA_KEY)

    @pytest.mark.parametrize("url", [PRIVATE_URL, GIST_URL])
    def test_agent_github_deposit_carries_no_operator_key(
        self, engine: TestClient, monkeypatch: pytest.MonkeyPatch, url: str
    ) -> None:
        seen = _record_github_requests(monkeypatch)
        token = _mint("agent", AGENT)
        resp = engine.post("/corpus/deposit/url", json={"url": url}, headers=_bearer(token))
        assert resp.status_code == 200, resp.text
        assert any(r.url.host == "api.github.com" for r in seen), seen
        assert not [str(r.url) for r in seen if _carries_operator_credential(r)]

    def test_agent_numista_coin_deposit_is_refused_without_spending_the_key(
        self, engine: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen = _record_github_requests(monkeypatch)
        token = _mint("agent", AGENT)
        resp = engine.post("/corpus/deposit/url", json={"url": NUMISTA_URL}, headers=_bearer(token))
        assert resp.status_code == 403, resp.text
        assert "NUMISTA_API_KEY" in resp.json()["detail"]
        assert not [str(r.url) for r in seen if _carries_operator_credential(r)]

    def test_operator_deposit_still_carries_the_key(
        self, engine: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The control: the operator's own deposit keeps its credential."""
        seen = _record_github_requests(monkeypatch)
        resp = engine.post("/corpus/deposit/url", json={"url": PRIVATE_URL})
        assert resp.status_code == 200, resp.text
        contents = [r for r in seen if "/contents/" in r.url.path]
        assert contents
        assert all(r.headers["authorization"] == f"Bearer {OPERATOR_GITHUB_KEY}" for r in contents)


class _LinkPostImporter:
    """A link-post importer whose post names a GitHub path (the follow's target)."""

    DEFAULT_FOLLOW_POST_LINKS = True

    def __init__(self, target: str) -> None:
        self.target = target

    def accepts_url(self, url: str) -> bool:
        return url.startswith("https://news.example.org/")

    async def deposit(
        self, session: Any, url: str, deposited_by: str, tags: list[str]
    ) -> tuple[str, str]:
        from particles.core.schema import FetchPolicy, Mutability, WarcRecordType
        from particles.corpus.deposit import save_blob, sha256, write_entry_and_snapshot

        content = json.dumps({"url": self.target}).encode()
        digest = sha256(content)
        return await write_entry_and_snapshot(
            session=session,
            uri_r=url,
            source_type="WEB_PAGE",
            mutability=Mutability.STABLE,
            fetch_policy=FetchPolicy.NEVER,
            content=content,
            content_hash=digest,
            archive_path=save_blob(content, digest),
            deposited_by=deposited_by,
            tags=tags,
            warc_record_type=WarcRecordType.RESPONSE,
        )

    def primary_url(self, content: bytes) -> str | None:
        return str(json.loads(content)["url"])


class TestFollowedLink:
    @pytest.mark.asyncio
    async def test_operator_deposit_follows_a_github_link_without_the_key(
        self, db_session: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The source's author chose the link, so the operator's deposit follows it anonymously."""
        from particles.corpus.deposit import deposit_url
        from particles.corpus.follow_edges import get_follow_sources
        from particles.ingest.importers import registry
        from particles.ingest.importers.github import GitHubImporter

        monkeypatch.setenv("GITHUB_API_KEY", OPERATOR_GITHUB_KEY)
        seen = _record_github_requests(monkeypatch)
        monkeypatch.setattr(
            registry,
            "get_importers",
            lambda: [_LinkPostImporter(PRIVATE_URL), GitHubImporter()],
        )
        targets: list[tuple[str, str, str]] = []
        await deposit_url(
            db_session,
            "https://news.example.org/item/1",
            deposited_by="operator",
            out_follow_targets=targets,
        )
        assert targets, "the follow did not run"
        assert await get_follow_sources(db_session, targets[0][0])
        assert any(r.url.host == "api.github.com" for r in seen)
        assert not [str(r.url) for r in seen if _carries_operator_credential(r)]
        # The scope ends with the follow: the operator's next fetch has its key.
        assert service_credential("github") == OPERATOR_GITHUB_KEY


class _RecordingImporter:
    """Records whether the operator's GitHub key was available to its refetch."""

    def __init__(self) -> None:
        self.saw: list[str | None] = []

    def accepts_url(self, url: str) -> bool:
        return url.startswith("https://github.com/")

    async def refetch(self, uri_r: str) -> bytes:
        self.saw.append(service_credential("github"))
        return b"refreshed"


async def _github_entry(session: Any, *, agent: bool) -> str:
    from contextlib import nullcontext

    from particles.core.schema import FetchPolicy, Mutability, WarcRecordType
    from particles.corpus.deposit import save_blob, sha256, write_entry_and_snapshot
    from particles.corpus.deposit_role import agent_deposit

    content = b"first"
    digest = sha256(content)
    with agent_deposit() if agent else nullcontext():
        entry_id, _snap = await write_entry_and_snapshot(
            session=session,
            uri_r=PRIVATE_URL,
            source_type="GITHUB_REPO",
            mutability=Mutability.MUTABLE,
            fetch_policy=FetchPolicy.LAZY,
            content=content,
            content_hash=digest,
            archive_path=save_blob(content, digest),
            deposited_by=AGENT if agent else "operator",
            tags=[],
            warc_record_type=WarcRecordType.RESPONSE,
        )
    return entry_id


class TestRefresh:
    @pytest.fixture(autouse=True)
    def _operator_secret(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GITHUB_API_KEY", OPERATOR_GITHUB_KEY)

    @pytest.mark.asyncio
    async def test_an_agents_entry_is_refreshed_without_the_key(self, db_session: Any) -> None:
        from particles.corpus.fetch import maybe_refetch

        entry_id = await _github_entry(db_session, agent=True)
        importer = _RecordingImporter()
        await maybe_refetch(db_session, entry_id, force=True, importers=[importer])
        assert importer.saw == [None]

    @pytest.mark.asyncio
    async def test_a_followed_entry_is_refreshed_without_the_key(self, db_session: Any) -> None:
        from particles.corpus.fetch import maybe_refetch
        from particles.corpus.follow_edges import add_follow_edge

        entry_id = await _github_entry(db_session, agent=False)
        await add_follow_edge(db_session, via_entry_id="post", target_entry_id=entry_id)
        importer = _RecordingImporter()
        await maybe_refetch(db_session, entry_id, force=True, importers=[importer])
        assert importer.saw == [None]

    @pytest.mark.asyncio
    async def test_the_operators_entry_is_refreshed_with_the_key(self, db_session: Any) -> None:
        from particles.corpus.fetch import maybe_refetch

        entry_id = await _github_entry(db_session, agent=False)
        importer = _RecordingImporter()
        await maybe_refetch(db_session, entry_id, force=True, importers=[importer])
        assert importer.saw == [OPERATOR_GITHUB_KEY]


#: Every kind of confined principal the URL deposit route serves: an agent key
#: (stamped and metered) and a harvester key (neither; security review on
#: PR 1535, S1).
_CONFINED_KEYS = [
    pytest.param(("agent", AGENT, {}), id="agent"),
    pytest.param(("agent", "claude-code-hook", {"reserved_ok": True}), id="harvester"),
]


class TestConfinedEntryRefresh:
    """Whatever confined principal deposited an entry, its refresh spends no operator key."""

    @pytest.mark.parametrize("key", _CONFINED_KEYS)
    def test_refresh_of_a_confined_deposit_carries_no_operator_key(
        self,
        engine: TestClient,
        monkeypatch: pytest.MonkeyPatch,
        key: tuple[str, str, dict[str, Any]],
    ) -> None:
        import asyncio

        from tests.test_engine_keys import HOOK_ROOTS

        monkeypatch.setenv("GITHUB_API_KEY", OPERATOR_GITHUB_KEY)
        seen = _record_github_requests(monkeypatch)
        role, principal, extra = key
        if extra.get("reserved_ok"):
            extra = {**extra, **HOOK_ROOTS}
        token = _mint(role, principal, **extra)
        resp = engine.post("/corpus/deposit/url", json={"url": PRIVATE_URL}, headers=_bearer(token))
        assert resp.status_code == 200, resp.text
        entry_id = resp.json()["entry_id"]
        seen.clear()

        async def _refresh() -> None:
            from particles.corpus.fetch import maybe_refetch
            from particles.db import get_engine, session_scope
            from particles.ingest.importers.registry import get_importers

            try:
                async with session_scope() as s:
                    await maybe_refetch(s, entry_id, force=True, importers=get_importers())
                    await s.commit()
            finally:
                await get_engine().dispose()

        asyncio.run(_refresh())
        assert any(r.url.host == "api.github.com" for r in seen), seen
        assert not [str(r.url) for r in seen if _carries_operator_credential(r)]

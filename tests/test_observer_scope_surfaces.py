# SPDX-FileCopyrightText: 2026 The Particles authors
#
# SPDX-License-Identifier: Apache-2.0

"""Observer scope on the session-facing surfaces (§3/§4).

The digest, the graph view, and a project-bound MCP server — its reads, its
explicit widening, and the rules that keep the project key the server's to name.
The candidate-selection sites themselves are in
``tests/test_query_negative_retrieval.py``.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from particles.config import get_config
from particles.db import DEFAULT_STORE
from particles.mcp.observer import bind_project
from tests._observer_scope import belief, project_source_tags, rescoped, source

A, B = "-proj-a", "-proj-b"


@pytest.fixture
def bound_to_a() -> Iterator[None]:
    bind_project(A)
    try:
        yield
    finally:
        bind_project(None)


async def _world(session: Any, *, rescope: bool = True) -> dict[str, str]:
    in_a = await source(session, "a", project_source_tags(A))
    in_b = await source(session, "b", project_source_tags(B))
    mine = await belief(session, "Project A deploys on Fridays.", in_a)
    theirs = await belief(session, "Project B deploys on Mondays.", in_b)
    world = await belief(
        session, "Pluto has five known moons.", await source(session, "p", ["web"])
    )
    if rescope:
        await rescoped(session)
    await session.commit()
    return {
        "mine": mine.id,
        "theirs": theirs.id,
        "world": world.id,
        "entry_a": in_a[0],
        "entry_b": in_b[0],
    }


async def _call(name: str, args: dict[str, Any]) -> Any:
    from particles.mcp import build_server

    result = await build_server().call_tool(name, args)
    return result[1] if isinstance(result, tuple) and len(result) == 2 else result


class TestDigest:
    async def test_a_project_digest_ranks_only_what_is_in_view_and_says_so(
        self, db_session: Any
    ) -> None:
        from particles.operations.digest import build_digest

        await _world(db_session)

        digest = await build_digest(DEFAULT_STORE, A)

        assert "Project A deploys on Fridays." in digest and "Pluto has five" in digest
        assert "Project B deploys" not in digest
        assert f"_Observer: project `{A}` — 2 of 3 ACTIVE belief(s) in scope" in digest
        assert "_2 of 2 ACTIVE belief(s), ranked by effective confidence._" in digest

    async def test_no_observer_is_the_digest_as_it_always_was(self, db_session: Any) -> None:
        from particles.operations.digest import build_digest

        await _world(db_session)

        digest = await build_digest(DEFAULT_STORE)

        assert "Project B deploys on Mondays." in digest
        assert "Observer" not in digest and "observer" not in digest

    async def test_an_observer_on_a_store_never_rescoped_stays_store_wide_and_says_so(
        self, db_session: Any
    ) -> None:
        from particles.operations.digest import build_digest

        await _world(db_session, rescope=False)

        digest = await build_digest(DEFAULT_STORE, A)

        assert "Project B deploys on Mondays." in digest  # nothing silently emptied
        assert f"Project observer `{A}` not applied" in digest
        assert "particles memory rescope" in digest


class TestBoundMcpServer:
    async def test_reads_go_through_the_bound_project(
        self, db_session: Any, bound_to_a: None
    ) -> None:
        w = await _world(db_session)

        listed = await _call("particles_list", {"status": "ACTIVE"})

        assert {p["id"] for p in listed["particles"]} == {w["mine"], w["world"]}
        assert listed["observer"] == {"project": A, "all_projects": False}

    async def test_widening_one_call_is_explicit_and_disclosed(
        self, db_session: Any, bound_to_a: None
    ) -> None:
        w = await _world(db_session)

        listed = await _call("particles_list", {"status": "ACTIVE", "all_projects": True})

        assert w["theirs"] in {p["id"] for p in listed["particles"]}
        assert listed["observer"] == {"project": A, "all_projects": True}

    async def test_a_belief_is_still_addressable_by_id(
        self, db_session: Any, bound_to_a: None
    ) -> None:
        """Reading by id is addressing, not retrieval: a contested badge's drill-down must work."""
        w = await _world(db_session)
        shown = await _call("particle_show", {"particle_id": w["theirs"]})
        assert w["theirs"] in str(shown) and "Project B deploys on Mondays." in str(shown)

    async def test_an_unbound_server_is_unchanged(self, db_session: Any) -> None:
        w = await _world(db_session)

        listed = await _call("particles_list", {"status": "ACTIVE"})

        assert w["theirs"] in {p["id"] for p in listed["particles"]}
        assert "observer" not in listed

    async def test_the_digest_resource_reads_through_the_bound_project(
        self, db_session: Any, bound_to_a: None
    ) -> None:
        from particles.mcp.resources import _render_digest

        await _world(db_session)

        assert "Project B deploys" not in await _render_digest(DEFAULT_STORE)
        assert "Project B deploys" in await _render_digest(DEFAULT_STORE, B)


def test_every_mcp_read_tool_has_decided_how_it_meets_the_observer() -> None:
    """A new read tool cannot skip the decision.

    A tool that selects beliefs by anything other than their id reads through
    the observer and takes ``all_projects``. One that addresses a record by id
    does not. One that returns maintenance records rather than beliefs does not.
    """
    import inspect

    from particles.mcp.server import _TOOL_REGISTRATION_ORDER

    filtered = {"query", "particles_list", "particle_search", "graph_view"}
    by_id = {"particle_show", "subjects_show", "event_show"}
    maintenance = {
        "subjects_list",
        "subjects_search",
        "list_taxonomies",
        "list_corpus_entries",
        "lint",
        "quality_report",
        "links_suggest",
        "corpus_links_suggest",
        "events_list",
    }
    names = {fn.__name__ for fn in _TOOL_REGISTRATION_ORDER}

    assert names == filtered | by_id | maintenance, "classify the new tool in this test"
    for fn in _TOOL_REGISTRATION_ORDER:
        takes_flag = "all_projects" in inspect.signature(fn).parameters
        assert takes_flag == (fn.__name__ in filtered), fn.__name__


class TestBoundMcpWrites:
    @pytest.fixture(autouse=True)
    def _writes_enabled(self) -> None:
        get_config().mcp.write.enabled_stores = [DEFAULT_STORE]

    async def _entry_tags(self, entry_id: str) -> list[str]:
        from particles.corpus.store import get_entry
        from particles.db import session_scope

        async with session_scope() as session:
            entry = await get_entry(session, entry_id)
        assert entry is not None
        return entry.tags

    async def test_a_deposit_is_stamped_with_the_servers_key_not_the_callers(
        self, db_session: Any, bound_to_a: None
    ) -> None:
        out = await _call("deposit_text", {"text": "notes", "tags": ["scratch", f"project:{B}"]})
        assert await self._entry_tags(out["corpus_entry_id"]) == ["scratch", f"project:{A}"]

    async def test_a_caller_cannot_name_a_project_on_an_unbound_server_either(
        self, db_session: Any
    ) -> None:
        out = await _call("deposit_text", {"text": "notes", "tags": [f"project:{B}"]})
        assert await self._entry_tags(out["corpus_entry_id"]) == []

    async def test_an_assertion_may_cite_its_own_projects_source_only(
        self, db_session: Any, bound_to_a: None
    ) -> None:
        from mcp.server.fastmcp.exceptions import ToolError

        w = await _world(db_session)
        args = {"content": "A claim.", "subject_names": ["x"], "confidence": 0.5}
        page = await source(db_session, "page2", ["web"])
        await db_session.commit()

        for foreign in (w["entry_b"], page[0]):  # another project's, and a global one
            with pytest.raises(ToolError, match="is not a source of project"):
                await _call("particle_assert", {**args, "corpus_entry_id": foreign})


class TestBoundMcpLinks:
    """A bound agent may link beliefs only within one observer scope.

    A ``CO_EVIDENTIAL`` edge unions the two groups' sources in the scope join,
    so a link across scopes would move a belief into another project's view.
    """

    @pytest.fixture(autouse=True)
    def _writes_enabled(self) -> None:
        get_config().mcp.write.enabled_stores = [DEFAULT_STORE]

    async def _link(self, a: str, b: str) -> Any:
        return await _call("link_add", {"particle_a": a, "particle_b": b})

    async def test_a_link_within_the_bound_project_is_written(
        self, db_session: Any, bound_to_a: None
    ) -> None:
        w = await _world(db_session)
        also_mine = await belief(
            db_session,
            "Project A deploys weekly.",
            await source(db_session, "a2", project_source_tags(A)),
        )
        await db_session.commit()
        out = await self._link(w["mine"], also_mine.id)
        assert out["verdict"] == "LINKED"

    @pytest.mark.parametrize("partner", ["theirs", "world", "unattributed"])
    async def test_a_link_across_scopes_is_refused(
        self, db_session: Any, bound_to_a: None, partner: str
    ) -> None:
        from mcp.server.fastmcp.exceptions import ToolError

        from particles.store.relation_store import get_relations_for_particle

        w = await _world(db_session)
        keyless = await belief(
            db_session, "A harvested note.", await source(db_session, "u", ["claude-code"])
        )
        await db_session.commit()
        other = keyless.id if partner == "unattributed" else w[partner]
        with pytest.raises(ToolError, match="particles links add"):
            await self._link(w["mine"], other)
        assert await get_relations_for_particle(db_session, w["mine"]) == []

    async def test_an_unbound_server_links_across_scopes(self, db_session: Any) -> None:
        w = await _world(db_session)
        assert (await self._link(w["mine"], w["theirs"]))["verdict"] == "LINKED"

    async def test_a_store_never_rescoped_is_not_checked(
        self, db_session: Any, bound_to_a: None
    ) -> None:
        w = await _world(db_session, rescope=False)
        assert (await self._link(w["mine"], w["world"]))["verdict"] == "LINKED"


def test_the_link_rule_holds_over_the_http_transport(
    cli_db: Any, monkeypatch: pytest.MonkeyPatch, bound_to_a: None
) -> None:
    """``HttpBackend`` carries the binding and ``POST /links`` enforces it."""
    import asyncio

    import httpx

    from particles.config import reset_config
    from particles.db import session_scope

    monkeypatch.setenv("PARTICLES_ENGINE_BASE_URL", "http://engine.test")
    reset_config()
    get_config().mcp.write.enabled_stores = [DEFAULT_STORE]

    from particles.api.app import app as fastapi_app

    recorded: list[str] = []

    async def _recorder(scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] == "http":
            recorded.append(scope["path"])
        await fastapi_app(scope, receive, send)

    real_async_client = httpx.AsyncClient

    def _asgi_client(**kwargs: Any) -> httpx.AsyncClient:
        return real_async_client(
            transport=httpx.ASGITransport(app=_recorder),
            base_url=kwargs["base_url"],
            headers=kwargs["headers"],
        )

    monkeypatch.setattr("particles.api.client.http.egress_client", _asgi_client)

    async def _seed() -> dict[str, str]:
        async with session_scope() as session:
            return await _world(session)

    w = asyncio.run(_seed())

    from mcp.server.fastmcp.exceptions import ToolError

    with pytest.raises(ToolError, match="particles links add"):
        asyncio.run(_call("link_add", {"particle_a": w["mine"], "particle_b": w["theirs"]}))
    assert "/links" in recorded


class TestGraphView:
    async def test_a_subjects_neighbourhood_is_read_through_the_observer(
        self, db_session: Any
    ) -> None:
        from particles.core.schema import Subject
        from particles.operations.graph_view import build_graph_data
        from particles.store.subject_store import insert_subject

        subject = Subject(canonical_name="deploys", asserted_by="t")
        await insert_subject(db_session, subject)
        in_a = await source(db_session, "a", project_source_tags(A))
        in_b = await source(db_session, "b", project_source_tags(B))
        mine = await belief(db_session, "A deploys on Fridays.", in_a, subject_ids=[subject.id])
        theirs = await belief(db_session, "B deploys on Mondays.", in_b, subject_ids=[subject.id])
        await rescoped(db_session)
        await db_session.commit()

        def particle_ids(graph: Any) -> set[str]:
            return {pid for pid in (mine.id, theirs.id) if pid in graph.model_dump_json()}

        scoped = await build_graph_data(db_session, subject_id=subject.id, observer_project=A)
        whole = await build_graph_data(db_session, subject_id=subject.id)

        assert particle_ids(scoped) == {mine.id}
        assert particle_ids(whole) == {mine.id, theirs.id}


class TestWidenedMcpReads:
    """Every filtered MCP read, unwidened and widened, on a project-bound server.

    Pins today's behaviour: ``all_projects=False`` sees global beliefs plus the
    bound project's and never another project's; ``all_projects=True`` also
    returns the other project's belief and discloses the widening in
    ``observer``. Widening is the caller's to lift, one call at a time.

    That the caller may lift it at all is an open question: the Atlas report
    lists caller-liftable widening under "Avoid", and removing it would mean
    amending the per-call widening rule. If that changes, change
    this test with it.
    """

    FINGERPRINT = "ab" * 32

    @pytest.fixture(autouse=True)
    def _no_model(self, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
        """The embedding model and the NL answer are the only seams replaced."""
        import particles.operations.query.main as qmain
        from particles import embeddings as ep

        model = MagicMock()
        model.encode = MagicMock(return_value=[np.ones(4, dtype=np.float32)])
        original = ep._embedding_model
        ep.set_embedding_model(model)
        monkeypatch.setattr(qmain, "_generate_response", AsyncMock(return_value="answer"))
        try:
            yield
        finally:
            ep.set_embedding_model(original)

    async def _world(self, session: Any) -> dict[str, str]:
        """One subject and one fingerprint shared by all three beliefs, so each
        tool's selector reaches every one of them and only the observer decides."""
        from particles.core.schema import Subject
        from particles.store.subject_store import insert_subject

        subject = Subject(canonical_name="deploys", asserted_by="t")
        await insert_subject(session, subject)
        shared = {"subject_ids": [subject.id], "context_fingerprint": self.FINGERPRINT}
        in_a = await source(session, "a", project_source_tags(A))
        in_b = await source(session, "b", project_source_tags(B))
        mine = await belief(session, "Project A deploys on Fridays.", in_a, **shared)
        theirs = await belief(session, "Project B deploys on Mondays.", in_b, **shared)
        world = await belief(
            session, "Pluto has five known moons.", await source(session, "p", ["web"]), **shared
        )
        await rescoped(session)
        await session.commit()
        return {"mine": mine.id, "theirs": theirs.id, "world": world.id, "subject": subject.id}

    @staticmethod
    def _args(tool: str, w: dict[str, str]) -> dict[str, Any]:
        match tool:
            case "query":
                return {"question": "When does each project deploy?"}
            case "particles_list":
                return {"status": "ACTIVE"}
            case "particle_search":
                return {"fingerprint": TestWidenedMcpReads.FINGERPRINT}
            case "graph_view":
                return {"subject_id": w["subject"]}
        raise AssertionError(tool)

    @staticmethod
    def _ids(tool: str, out: dict[str, Any]) -> set[str]:
        if tool == "graph_view":
            return set(out["particles"])
        return {p["id"] for p in out["particles"]}

    @pytest.mark.parametrize("tool", ["query", "particles_list", "particle_search", "graph_view"])
    async def test_widening_crosses_projects_and_says_so(
        self, db_session: Any, bound_to_a: None, tool: str
    ) -> None:
        w = await self._world(db_session)
        args = self._args(tool, w)

        unwidened = await _call(tool, args)
        widened = await _call(tool, {**args, "all_projects": True})

        assert self._ids(tool, unwidened) == {w["mine"], w["world"]}
        assert unwidened["observer"] == {"project": A, "all_projects": False}
        assert self._ids(tool, widened) == {w["mine"], w["world"], w["theirs"]}
        assert widened["observer"] == {"project": A, "all_projects": True}

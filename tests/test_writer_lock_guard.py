# SPDX-FileCopyrightText: 2026 The Particles authors
# SPDX-License-Identifier: Apache-2.0
"""No writer calls a model, a live authority or the embedder while it holds the writer lock.

This guards :func:`particles.db.write_lock`'s contract: it wraps a write
transaction, never the LLM / fetch / embed phase, as
``tests/test_egress_chokepoint.py`` guards the transports. Every writer runs
gather / apply (D2): the subject lookups, the embedding and every
model probe before the lock, the writes under it (security review 2026-10-31-1,
the writer-lock class).

Each row drives one writer on a scratch file-backed store, configured only
through the loader (``network.egress: none``, live authorities on for every
source type), with three stubs that record whether a store's writer lock, or
SQLite's own write lock (seen from a second connection), is held when they are
called: the model (``particles.llm.complete``), a live subject authority, and
the embedder. A recorder on every connection also notes, for each statement
that writes a ``particles`` row and for the commit that follows it, whether the
writer lock of the store that connection writes was held. A row fails when any
stub was called with a lock held, when a particle was written or committed with
the store's writer lock free, or when a stub it names was never called (the row
would then guard nothing). A belief write is every row of it: the particle, its
subjects and subject links, its provenance edges and its operator events
(security review 2026-10-31-3, AF18).

A belief writer's row also fails unless its apply ran the conflict search
against the store as it stands, under the lock
(``pipeline.search_store_under_lock``, security review 2026-10-31-2, AE15),
and then read every candidate that search returned: an apply that searched and
decided on something else guards nothing (security review 2026-10-31-3, AF18).
A new writer is one more row.
"""

from __future__ import annotations

import asyncio
import re
import sqlite3
from collections.abc import Callable, Coroutine, Generator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi.testclient import TestClient

LEGACY = "writer-lock-guard-legacy-operator-key"
VECTOR = [0.1, 0.2, 0.3, 0.4]


class _SearchResult(list[Any]):
    """A search's candidates, noting when the caller has read every one of them."""

    def __init__(self, items: Any, on_read: Callable[[], None]) -> None:
        super().__init__(items)
        self._on_read = on_read
        if not self:
            on_read()

    def __iter__(self) -> Any:
        yield from super().__iter__()
        self._on_read()


@dataclass
class _Search:
    """One conflict search: whether it ran under the lock, read the store's
    changes, and had its candidates read by the apply."""

    locked: bool
    found: bool
    read: bool = False


@dataclass
class Recorder:
    """One list per stub: whether a lock was held at each call; and per particle write."""

    model: list[bool] = field(default_factory=list)
    authority: list[bool] = field(default_factory=list)
    embed: list[bool] = field(default_factory=list)
    writes: list[bool] = field(default_factory=list)
    """Whether the written store's writer lock was held at each statement that
    wrote a ``particles`` row."""
    commits: list[bool] = field(default_factory=list)
    """Whether it was held at each commit of a transaction that wrote one."""
    search_log: list[_Search] = field(default_factory=list)

    @property
    def searches(self) -> list[bool]:
        """Whether each conflict search of the store ran under the lock, read the
        store's changes, and had every candidate it returned read by its apply."""
        return [s.locked and s.found and s.read for s in self.search_log]

    def held(self) -> dict[str, int]:
        """Every breach of the lock contract, by kind: a stub called with a lock
        held, a particle written or committed with the writer lock free."""
        breaches = {
            kind: sum(calls)
            for kind, calls in (
                ("model", self.model),
                ("authority", self.authority),
                ("embed", self.embed),
            )
            if any(calls)
        }
        for kind, calls in (("unlocked write", self.writes), ("unlocked commit", self.commits)):
            if not all(calls):
                breaches[kind] = calls.count(False)
        return breaches


#: The SQLite files of the guard's stores, set by the ``recorder`` fixture.
_STORE_FILES: list[Path] = []


def _sqlite_write_locked(path: Path) -> bool:
    """Whether another connection holds SQLite's write lock on ``path``."""
    if not path.exists():
        return False
    con = sqlite3.connect(path, timeout=0)
    try:
        con.execute("BEGIN IMMEDIATE")
        con.execute("ROLLBACK")
        return False
    except sqlite3.OperationalError:
        return True
    finally:
        con.close()


def _held() -> bool:
    """Whether any store's writer lock, or SQLite's write lock on its file, is held."""
    from particles.db import _engines, write_lock_held

    if any(write_lock_held(handle) for handle in list(_engines)) or write_lock_held("default"):
        return True
    return any(_sqlite_write_locked(path) for path in _STORE_FILES)


_PARTICLE_WRITE = re.compile(
    r"^\s*(?:INSERT\s+(?:OR\s+\w+\s+)?INTO|UPDATE|DELETE\s+FROM)\s+"
    r"\"?(?:particles|subjects|particle_subjects|particle_provenance_edges|operator_events)\"?"
    r"[\s(]",
    re.IGNORECASE,
)


def _store_of_connection(conn: Any) -> str:
    """The store whose engine ``conn`` belongs to, else the default store."""
    from particles.db import DEFAULT_STORE, _engines

    for handle, engine in list(_engines.items()):
        if engine.sync_engine is conn.engine:
            return handle
    return DEFAULT_STORE


def _record_particle_writes(rec: Recorder) -> Callable[[], None]:
    """Record the writer-lock state at each particle write and at its commit.

    Returns the function that removes the listeners.
    """
    from sqlalchemy import event
    from sqlalchemy.engine import Engine

    from particles.db import write_lock_held

    def _execute(conn: Any, _cursor: Any, statement: str, *_a: Any) -> None:
        if _PARTICLE_WRITE.match(statement):
            rec.writes.append(write_lock_held(_store_of_connection(conn)))
            conn.info["guard_particle_write"] = True

    def _commit(conn: Any) -> None:
        if conn.info.pop("guard_particle_write", False):
            rec.commits.append(write_lock_held(_store_of_connection(conn)))

    def _rollback(conn: Any) -> None:
        conn.info.pop("guard_particle_write", None)

    listeners = (
        ("before_cursor_execute", _execute),
        ("commit", _commit),
        ("rollback", _rollback),
    )
    for name, fn in listeners:
        event.listen(Engine, name, fn)

    def _remove() -> None:
        for name, fn in listeners:
            event.remove(Engine, name, fn)

    return _remove


class _RecordingAuthority:
    """A live authority that finds nothing and records the lock state per lookup."""

    NAMESPACE = "guard"
    PRIORITY = 0
    LIVE = True
    APPLICABILITY: list[Any] = []

    def __init__(self, recorder: Recorder) -> None:
        self.recorder = recorder

    def uri_for(self, external_id: str) -> str | None:
        return None

    def recognize(self, name: str) -> Any:
        return None

    async def resolve(
        self, session: Any, name: str, *, particle_content: str | None, domain: str | None
    ) -> Any:
        self.recorder.authority.append(_held())
        return None

    async def canonical_name_for(self, session: Any, external_id: str) -> str | None:
        return None


def _create_tables(store: str = "default") -> None:
    async def _go() -> None:
        import particles._orm_modules  # noqa: F401
        from particles.db import Base, get_engine

        async with get_engine(store).begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        await get_engine(store).dispose()

    asyncio.run(_go())


@pytest.fixture
def recorder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Generator[Recorder, None, None]:
    import particles.ingest.subject_resolver as sr
    import particles.llm as llm_pkg
    from particles import embeddings as ep
    from particles.api import _keys
    from particles.config import reset_config

    home = tmp_path / "home"
    home.mkdir()
    keydir = tmp_path / "keys"
    keydir.mkdir(mode=0o700)
    dest = tmp_path / "dest.db"
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "network:\n  egress: none\n"
        "mcp:\n  write:\n    enabled_stores: [default]\n"
        "subjects:\n  skip_live_authorities_source_types: []\n"
        f"storage:\n  stores:\n    dest: sqlite+aiosqlite:///{dest}\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("PARTICLES_CONFIG", str(cfg))
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path / 'guard.db'}")
    monkeypatch.setenv("PARTICLES_BLOB_DIR", str(tmp_path / "blobs"))
    monkeypatch.setenv("PARTICLES_API_KEYS_FILE", str(keydir / "engine-keys.json"))
    monkeypatch.setenv("PARTICLES_API_KEY", LEGACY)
    monkeypatch.delenv("CLAUDECODE", raising=False)
    reset_config()
    _keys.reset_key_cache()
    _keys.forget_key_files()
    _create_tables("default")
    _create_tables("dest")

    rec = Recorder()
    _STORE_FILES[:] = [tmp_path / "guard.db", dest]
    remove_listeners = _record_particle_writes(rec)
    import particles.ingest.pipeline as pipeline
    from particles.db import store_of, write_lock_held

    real_search = pipeline.search_store_under_lock

    async def _search(session: Any, *a: Any, **k: Any) -> Any:
        found = await real_search(session, *a, **k)
        entry = _Search(locked=write_lock_held(store_of(session)), found=found is not None)
        rec.search_log.append(entry)
        if found is None:
            return None

        def _read() -> None:
            entry.read = True

        return _SearchResult(found[0], _read), found[1]

    monkeypatch.setattr(pipeline, "search_store_under_lock", _search)

    async def _complete(*_a: Any, **_k: Any) -> str:
        rec.model.append(_held())
        return "VERDICT: NO"

    def _encode(texts: Any, *_a: Any, **_k: Any) -> Any:
        rec.embed.append(_held())
        count = len(texts) if isinstance(texts, list) else 1
        return np.array([VECTOR] * count, dtype=np.float32)

    monkeypatch.setattr(llm_pkg, "complete", _complete)
    authority = _RecordingAuthority(rec)
    monkeypatch.setattr(sr, "get_authorities", lambda: [authority])
    model = MagicMock()
    model.encode = MagicMock(side_effect=_encode)
    original = ep._embedding_model
    ep.set_embedding_model(model)
    yield rec
    remove_listeners()
    _STORE_FILES.clear()
    ep.set_embedding_model(original)
    _keys.reset_key_cache()
    _keys.forget_key_files()
    reset_config()


def _run(coro: Coroutine[Any, Any, Any]) -> Any:
    from particles.db import reset_engine

    try:
        return asyncio.run(coro)
    finally:
        reset_engine()


def _client() -> TestClient:
    from particles.api.app import app

    return TestClient(app, base_url="http://127.0.0.1:8000", client=("127.0.0.1", 50000))


def _agent_headers() -> dict[str, str]:
    from particles.api import _keys
    from particles.api._keys import Role

    return {"Authorization": f"Bearer {_keys.mint(Role.AGENT, 'agent:guard')[0]}"}


_OPERATOR = {"Authorization": f"Bearer {LEGACY}"}


async def _local_assert(content: str, names: list[str] | None = None) -> Any:
    from particles.api.client.local import LocalBackend

    return await LocalBackend().particle_assert(
        content=content,
        subject_names=["Guard Widget"] if names is None else names,
        confidence=0.7,
        source_excerpt=content,
        corpus_entry_id=None,
        uncertainty_nature="EPISTEMIC",
        tags=None,
        store="default",
    )


def _seed(content: str, names: list[str] | None = None) -> Any:
    return _run(_local_assert(content, names))


def _post(path: str, body: dict[str, Any], headers: dict[str, str]) -> None:
    resp = _client().post(path, json=body, headers=headers)
    assert resp.status_code < 300, (path, resp.status_code, resp.text)


def _belief(content: str) -> dict[str, Any]:
    return {
        "content": content,
        "subject_names": ["Guard Widget"],
        "confidence": 0.7,
        "source_excerpt": content,
    }


# --------------------------------------------------------------------------- #
# The writers
# --------------------------------------------------------------------------- #


def _http_agent_assert() -> None:
    headers = _agent_headers()
    _post("/particles/assert", _belief("The guard widget ships in blue."), headers)
    _post("/particles/assert", _belief("The guard widget ships in red."), headers)


def _http_agent_supersede() -> None:
    headers = _agent_headers()
    resp = _client().post(
        "/particles/assert", json=_belief("The guard widget ships in blue."), headers=headers
    )
    assert resp.status_code < 300, resp.text
    # A second near claim, so the successor's probe has a pair to ask about.
    _post("/particles/assert", _belief("The guard widget ships in red."), headers)
    body = _belief("The guard widget ships in green.")
    body["supersedes_id"] = resp.json()["asserted_particle_id"]
    _post("/particles/supersede", body, headers)


def _http_operator_supersede() -> None:
    first = _seed("The guard widget ships in blue.")
    _seed("The guard widget ships in red.")
    body = _belief("The guard widget ships in grey.")
    body["reason"] = "the catalogue changed"
    _post(f"/particles/{first.asserted_particle_id}/supersede", body, _OPERATOR)


def _http_assign_subject() -> None:
    orphan = _seed("The guard widget ships in blue.", names=[])
    _post(
        f"/particles/{orphan.asserted_particle_id}/subjects",
        {"subject_name": "Guard Gadget"},
        _OPERATOR,
    )


async def _subject_of(particle_id: str) -> str:
    from particles.db import session_scope
    from particles.store.particle_store import get_particle

    async with session_scope() as session:
        particle = await get_particle(session, particle_id)
        assert particle is not None and particle.subject_ids
        return particle.subject_ids[0]


def _http_subject_split() -> None:
    first = _seed("The guard widget ships in blue.")
    source = _run(_subject_of(first.asserted_particle_id))
    _post(
        f"/subjects/{source}/split",
        {"particle_ids": [first.asserted_particle_id], "new_name": "Guard Gadget"},
        _OPERATOR,
    )


def _local_backend_assert_and_supersede() -> None:
    from particles.api.client.local import LocalBackend

    async def _go() -> None:
        first = await _local_assert("The guard widget ships in blue.")
        await _local_assert("The guard widget ships in red.")
        await LocalBackend().particle_supersede(
            supersedes_id=first.asserted_particle_id,
            content="The guard widget ships in green.",
            subject_names=["Guard Widget"],
            confidence=0.7,
            source_excerpt="it ships in green",
            corpus_entry_id=None,
            uncertainty_nature="EPISTEMIC",
            tags=None,
            store="default",
            reason="repainted",
        )

    _run(_go())


def _local_backend_subject_split() -> None:
    from particles.api.client.local import LocalBackend

    first = _seed("The guard widget ships in blue.")
    source = _run(_subject_of(first.asserted_particle_id))
    _run(
        LocalBackend().subject_split(
            source_id=source,
            particle_ids=[first.asserted_particle_id],
            new_name="Guard Gadget",
            new_external_id=None,
            dry_run=False,
        )
    )


def _memory_facade(name: str, arguments: dict[str, Any]) -> Callable[[], None]:
    def _drive() -> None:
        import mcp.types as types

        from particles.mcp.memory_compat import server

        async def _go() -> Any:
            srv = server.build_server("default")
            call = srv.request_handlers[types.CallToolRequest]
            return await call(
                types.CallToolRequest(
                    method="tools/call",
                    params=types.CallToolRequestParams(name=name, arguments=arguments),
                )
            )

        if name == "add_observations":
            _memory_facade(
                "create_entities",
                {"entities": [{"name": "Widget", "entityType": "product", "observations": []}]},
            )()
        result = _run(_go())
        assert not getattr(result.root, "isError", False), result

    return _drive


def _extraction() -> None:
    from particles.core.schema import Mutability, UncertaintyNature
    from particles.extraction.general import CandidateParticle, ExtractionResult

    class _Extractor:
        """One claim per snapshot, its year read from the snapshot's text."""

        EXTRACTOR_ID = "guard-stub"
        EXTRACTOR_VERSION = "0.0.1"

        def accepts(self, source_type: str) -> bool:
            return True

        async def extract(self, snapshot: Any, content: bytes, **kwargs: object) -> Any:
            year = content.decode().split()[-1].rstrip(".")
            return ExtractionResult(
                candidates=[
                    CandidateParticle(
                        content=f"The guard compressor was released in {year}.",
                        confidence_value=0.9,
                        uncertainty_nature=UncertaintyNature.EPISTEMIC,
                        subjects=["Guard Compressor", "Guard Maker"],
                    )
                ]
            )

    async def _deposit_and_extract(text: str) -> None:
        from particles.corpus.deposit import deposit_text_versioned
        from particles.db import session_scope
        from particles.ingest.pipeline import extract_snapshot

        async with session_scope() as session:
            entry_id, snapshot_id, _same = await deposit_text_versioned(
                session,
                text=text,
                uri_r="https://example.invalid/guard",
                source_type="LOCAL_MARKDOWN",
                mutability=Mutability.APPEND_ONLY,
                tags=[],
            )
            await session.commit()
            await extract_snapshot(session, entry_id, snapshot_id, extractor=_Extractor())
            await session.commit()

    _run(_deposit_and_extract("The guard compressor shipped in 2019."))
    # The second pass meets the first pass's claim, so its probe runs.
    _run(_deposit_and_extract("The guard compressor shipped in 2019. It shipped in 2020."))


def _interchange_import(tmp_path: Path) -> None:
    from particles.api.cli import interchange as cli_interchange
    from particles.db import session_scope
    from particles.interchange.store import export_store_bundle

    _seed("The guard widget ships in blue.")
    _seed("The guard widget ships in red.")

    async def _export() -> dict[str, str]:
        async with session_scope("default") as session:
            return await export_store_bundle(session)

    files = _run(_export())
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    for name, content in files.items():
        (bundle / name).write_text(content, encoding="utf-8")
    _run(cli_interchange._import(bundle, "default"))


def _curate_apply_supersede() -> None:
    from particles.api.cli import curate
    from particles.operations.curation.session import BeliefRevision

    first = _seed("The guard widget ships in blue.")
    _seed("The guard widget ships in red.")
    _run(
        curate._apply(
            gesture="supersede",
            card_key=f"stale:{first.asserted_particle_id}",
            reason="the catalogue changed",
            days=None,
            revision=BeliefRevision(content="The guard widget ships in green.", confidence=0.8),
        )
    )


# --------------------------------------------------------------------------- #
# The writers that return a held claim to the store (security review
# 2026-10-31-3, AF1): each seeds a record (A, quarantined B) and a claim C near
# B committed while B was held, every seed write under the lock.
# --------------------------------------------------------------------------- #


def _claim(content: str, **fields: Any) -> Any:
    from particles.core.schema import (
        Confidence,
        Particle,
        ProvenanceRef,
        ProvenanceRefType,
        UncertaintyNature,
    )
    from particles.core.scoring.confidence import CalibrationSource

    entry = fields.pop("entry", "guard-entry")
    return Particle(
        content=content,
        confidence=Confidence(value=0.8, calibration_source=CalibrationSource.EXTRACTOR_DIRECT),
        uncertainty_nature=UncertaintyNature.EPISTEMIC,
        asserted_by="general-extractor",
        provenance=[
            ProvenanceRef(
                type=ProvenanceRefType.SOURCE, corpus_entry_id=entry, snapshot_id=f"{entry}-s"
            )
        ],
        **fields,
    )


def _seed_held(retire_a: bool = False) -> tuple[str, str, str]:
    """``(record id, B id, C id)``: the record, then C committed while B is held."""
    from particles.core.conflict_resolution import build_inconsistency_particle
    from particles.core.status import Status, StatusReason
    from particles.db import session_scope, write_transaction
    from particles.store.particle_store import insert_particle, update_particle_status

    a = _claim("The guard widget ships in blue.", status=Status.ACTIVE, entry="guard-entry-a")
    b = _claim(
        "The guard widget ships only in red.",
        status=Status.PROVENANCE_STALE,
        status_reason=StatusReason.CONFLICT_PENDING,
        entry="guard-entry-b",
    )
    record = build_inconsistency_particle(
        a, b, corpus_entry_id="guard-entry-b", snapshot_id="guard-entry-b-s"
    )
    c = _claim("The guard widget ships only in green.", status=Status.ACTIVE)

    async def _go() -> None:
        async with session_scope() as session, write_transaction(session):
            await insert_particle(session, a, VECTOR)
            await insert_particle(session, b, VECTOR)
            await insert_particle(session, record)
            if retire_a:
                await update_particle_status(
                    session, a.id, Status.PROVENANCE_STALE, StatusReason.SUPERSEDED_BY_UPDATE
                )
        async with session_scope() as session, write_transaction(session):
            await insert_particle(session, c, VECTOR)

    _run(_go())
    return record.id, b.id, c.id


def _http_review_prefer_b() -> None:
    record_id, _b, _c = _seed_held()
    _post(f"/review/{record_id}", {"action": "PREFER_B", "reviewer_id": "guard"}, _OPERATOR)


def _trust_cascade() -> None:
    from particles.core.schema import (
        PolicyProvenance,
        SourceRef,
        SourceRefType,
        SourceTrustStatement,
    )
    from particles.core.status import Status
    from particles.db import session_scope, write_transaction
    from particles.operations.trust import set_trust_statement
    from particles.store.particle_store import ParticleRow, insert_particle
    from tests.test_cascade import _seed_quarantined_conflict

    async def _go() -> int:
        async with session_scope() as session:
            async with write_transaction(session):
                _a, b, _inc = await _seed_quarantined_conflict(session)
                row = await session.get(ParticleRow, b.id)  # type: ignore[attr-defined]
                stored = ParticleRow.from_model(b, VECTOR)  # type: ignore[arg-type]
                row.embedding_json = stored.embedding_json
                row.embedding_model_id = stored.embedding_model_id
            async with write_transaction(session):
                await insert_particle(
                    session,
                    _claim("Weight: 3.9 g (web)", status=Status.ACTIVE),
                    VECTOR,
                )
            stmt = SourceTrustStatement(
                domain="numismatics",
                source_ref=SourceRef(
                    type=SourceRefType.CORPUS_ENTRY,
                    value=b.provenance[0].corpus_entry_id,  # type: ignore[attr-defined]
                ),
                trust_rank=0.95,
                policy_provenance=PolicyProvenance.OPERATOR_DIRECT,
                asserted_by="operator",
            )
            # The trust endpoints run the cascade inside the caller's writer lock.
            async with write_transaction(session):
                return await set_trust_statement(session, stmt, actor="guard")

    assert _run(_go()) == 1


def _release_pass() -> None:
    from particles.db import session_scope
    from particles.operations.held_release import run_release

    _seed_held(retire_a=True)

    async def _go() -> Any:
        async with session_scope() as session:
            return await run_release(session)

    report = _run(_go())
    assert report.examined == 1, report


def _disclosure_withdrawal() -> None:
    from unittest.mock import AsyncMock, patch

    from particles.db import session_scope
    from particles.ingest.second_reading import ProbeVerdict
    from particles.operations.contradiction_disclosure import run_disclosure

    _seed_held()
    verdict = ProbeVerdict(usable=True, contradicts=False, description="not the same slot")

    async def _go() -> Any:
        async with session_scope() as session:
            return await run_disclosure(session, confirmed=[], mint=False, reread=True)

    with patch(
        "particles.ingest.second_reading._llm_verify_contradiction",
        AsyncMock(return_value=verdict),
    ):
        report = _run(_go())
    assert report.extract_reread["withdrawn"] == 1, report.extract_reread


def _reanchor_restatement() -> None:
    from unittest.mock import patch

    import particles.ingest.pipeline as pipeline
    from particles.core.status import Status, StatusReason
    from particles.db import session_scope, write_transaction
    from particles.operations.reanchor import run_reanchor
    from particles.store.particle_store import update_particle_status
    from tests.test_reanchor import (
        RESIDENCE,
        SESSION_4,
        SESSION_7,
        _one,
        _Scripted,
        _session,
        _Sessions,
    )

    async def _go() -> Any:
        ex = _Sessions()
        async with session_scope() as session:
            await _session(session, ex, SESSION_4, day=4)
            await _session(session, ex, SESSION_7, day=20)
            residence = await _one(session, RESIDENCE)
            async with write_transaction(session):
                await update_particle_status(
                    session,
                    residence.id,
                    Status.PROVENANCE_STALE,
                    StatusReason.SUPERSEDED_BY_UPDATE,
                )
            # The seeding extraction searches too: count only the pass's own.
            with patch.object(pipeline, "search_store_under_lock", _counted), _Scripted():
                return await run_reanchor(session, cursor=None)

    searched = pipeline.search_store_under_lock
    calls: list[bool] = []

    async def _counted(session: Any, *a: Any, **k: Any) -> Any:
        calls.append(True)
        return await searched(session, *a, **k)

    report = _run(_go())
    assert report.restated, report
    assert calls, "the re-anchor pass wrote its restatement with no search under the lock"


@dataclass(frozen=True)
class Writer:
    name: str
    drive: Callable[..., None]
    reaches: frozenset[str]
    """The stubs the writer must call, so the row is known to exercise it."""
    needs_tmp: bool = False
    searches: bool = True
    """A belief writer: its apply must search the store again under the lock."""


ALL = frozenset({"model", "authority", "embed"})
WRITERS = [
    Writer("http agent assert", _http_agent_assert, ALL),
    Writer("http agent supersede", _http_agent_supersede, ALL),
    Writer("http operator supersede", _http_operator_supersede, ALL),
    Writer(
        "http assign-subject",
        _http_assign_subject,
        frozenset({"authority", "embed"}),
        searches=False,
    ),
    Writer(
        "http subject split", _http_subject_split, frozenset({"authority", "embed"}), searches=False
    ),
    Writer("local backend assert and supersede", _local_backend_assert_and_supersede, ALL),
    Writer(
        "local backend subject split",
        _local_backend_subject_split,
        frozenset({"authority"}),
        searches=False,
    ),
    Writer(
        "memory facade create_entities",
        _memory_facade(
            "create_entities",
            {
                "entities": [
                    {
                        "name": "Pluto",
                        "entityType": "planet",
                        "observations": ["Pluto orbits beyond Neptune", "Pluto has five moons"],
                    }
                ]
            },
        ),
        frozenset({"model", "embed"}),
    ),
    Writer(
        "memory facade add_observations",
        _memory_facade(
            "add_observations",
            {
                "observations": [
                    {
                        "entityName": "Widget",
                        "contents": ["The widget ships in red.", "The widget ships in green."],
                    }
                ]
            },
        ),
        frozenset({"model", "embed"}),
    ),
    Writer("extraction write phase", _extraction, ALL),
    Writer("interchange import", _interchange_import, frozenset({"model", "embed"}), True),
    Writer("curate apply supersede", _curate_apply_supersede, ALL),
    Writer("http review prefer_b", _http_review_prefer_b, frozenset()),
    Writer("trust cascade", _trust_cascade, frozenset()),
    Writer("consolidation release pass", _release_pass, frozenset({"model", "embed"})),
    Writer("consolidation disclosure withdrawal", _disclosure_withdrawal, frozenset()),
    Writer(
        "consolidation re-anchor restatement",
        _reanchor_restatement,
        frozenset({"authority", "embed"}),
    ),
]


@pytest.mark.parametrize("writer", WRITERS, ids=[w.name for w in WRITERS])
def test_no_writer_calls_a_model_an_authority_or_the_embedder_under_the_writer_lock(
    writer: Writer, recorder: Recorder, tmp_path: Path
) -> None:
    """Every writer gathers before the store's writer lock and applies under it."""
    if writer.needs_tmp:
        writer.drive(tmp_path)
    else:
        writer.drive()
    reached = {
        kind
        for kind, calls in (
            ("model", recorder.model),
            ("authority", recorder.authority),
            ("embed", recorder.embed),
        )
        if calls
    }
    assert writer.reaches <= reached, (
        f"{writer.name} never reached {sorted(writer.reaches - reached)}; the row guards nothing"
    )
    assert recorder.writes, f"{writer.name} wrote no particle; the row guards nothing"
    assert recorder.held() == {}, f"{writer.name} broke the writer-lock contract: {recorder.held()}"


@pytest.mark.parametrize(
    "writer", [w for w in WRITERS if w.searches], ids=[w.name for w in WRITERS if w.searches]
)
def test_every_belief_writer_searches_the_store_again_under_the_writer_lock(
    writer: Writer, recorder: Recorder, tmp_path: Path
) -> None:
    """Every belief writer's apply runs its conflict search against the store as it
    stands, with the writer lock held, never only against what it read before the lock."""
    if writer.needs_tmp:
        writer.drive(tmp_path)
    else:
        writer.drive()
    assert recorder.searches, f"{writer.name} never searched the store under the writer lock"
    assert all(recorder.searches), (
        f"{writer.name} ran a conflict search with the lock free or unable to read the store's "
        f"changes: {recorder.searches}"
    )


def test_the_lock_seam_sees_the_writer_lock(recorder: Recorder) -> None:
    """Control: ``write_lock_held`` is True inside ``write_lock`` and False outside it."""
    from particles.db import write_lock, write_lock_held

    async def _go() -> tuple[bool, bool, bool]:
        before = write_lock_held("default")
        async with write_lock("default"):
            inside = write_lock_held("default")
        return before, inside, write_lock_held("default")

    assert _run(_go()) == (False, True, False)


# --------------------------------------------------------------------------- #
# The guard's own blind spots (security review 2026-10-31-3, AF18)
# --------------------------------------------------------------------------- #


def test_the_search_column_fails_when_an_apply_decides_on_less_than_its_search_read(
    recorder: Recorder, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A belief writer whose apply runs the search under the lock and then decides on
    none of what it read fails the search column."""
    import particles.ingest.pipeline as pipeline

    recorded = pipeline.search_store_under_lock  # the guard's own wrapper
    discarded: list[int] = []

    async def _discard(session: Any, *a: Any, **k: Any) -> Any:
        found = await recorded(session, *a, **k)
        if found is None:
            return None
        discarded.append(len(found[0]))
        return [], found[1]

    monkeypatch.setattr(pipeline, "search_store_under_lock", _discard)
    _local_backend_assert_and_supersede()
    assert any(n > 0 for n in discarded), discarded
    assert not (recorder.searches and all(recorder.searches))


def test_the_guard_fails_when_a_belief_write_commits_a_subject_with_the_lock_free(
    recorder: Recorder, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A belief writer that commits a Subject with the writer lock free fails its row."""
    import particles.ingest.subject_resolver as sr
    from particles.core.schema import Subject
    from particles.db import write_lock_held
    from particles.store.subject_store import insert_subject

    real = sr.prefetch_subjects
    outside: list[bool] = []

    async def _mint_before_lock(session: Any, *a: Any, **k: Any) -> Any:
        outside.append(write_lock_held("default"))
        await insert_subject(
            session, Subject(canonical_name=f"Lockless {len(outside)}", asserted_by="x")
        )
        await session.commit()
        return await real(session, *a, **k)

    monkeypatch.setattr(sr, "prefetch_subjects", _mint_before_lock)
    _local_backend_assert_and_supersede()
    assert outside and not any(outside), outside
    assert recorder.held() != {}

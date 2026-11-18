# SPDX-FileCopyrightText: 2026 The Particles authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: F811 (the engine and keys_file fixtures are imported, then named as parameters)

"""An agent key's deposits are bounded in the extraction queue.

Covers ``particles/corpus/metering.py``: the mark a metered deposit leaves on
its entry, the cap on a principal's unclaimed snapshots, and the ``429`` the
deposit routes answer with. The steady state at the end runs twenty simulated
days of deposits and nightly pass-1 selections against a scratch store, with
no model call: extraction is simulated by marking the chosen snapshots read.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from particles.config import get_config
from particles.core.schema import ExtractionStatus, Mutability
from particles.corpus.deposit import (
    deposit_text,
    deposit_text_versioned,
    own_entries_only_for,
    writes_committed_under_lock,
)
from particles.corpus.metering import (
    PendingExtractionsFull,
    metered_deposits_for,
    unclaimed_snapshot_count,
)
from particles.corpus.store import (
    CorpusEntryRow,
    SnapshotRow,
    list_pending_snapshots_for_catchup,
)
from particles.operations import consolidation as consolidation_mod
from tests.test_engine_keys import (  # noqa: F401
    HOOK,
    HOOK_ROOTS,
    _bearer,
    _mint,
    engine,
    keys_file,
)
from tests.test_write_route_gate import OPEN

AGENT = "agent:laptop-mcp"
OTHER = "agent:ci-bot"


@pytest.fixture(autouse=True)
def _blobs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PARTICLES_BLOB_DIR", str(tmp_path / "blobs"))


@contextlib.contextmanager
def _as(principal: str, *, metered: bool = True) -> Iterator[None]:
    """A deposit made as an agent key's route makes it: confined, and metered unless told not."""
    with own_entries_only_for(principal):
        if metered:
            with metered_deposits_for(principal):
                yield
        else:
            yield


async def _deposit(session: AsyncSession, text: str, principal: str = AGENT, **kw: Any) -> str:
    with _as(principal, **kw):
        entry_id, _ = await deposit_text(session, text, principal, "CONVERSATION", [])
    await session.flush()
    return entry_id


async def _mark(session: AsyncSession, entry_id: str) -> str | None:
    row = await session.get(CorpusEntryRow, entry_id)
    assert row is not None
    await session.refresh(row)
    return row.metered_principal


async def _claim_one(session: AsyncSession, entry_id: str) -> None:
    """Simulate pass 1 claiming a snapshot and handing it back PENDING."""
    await session.execute(
        update(SnapshotRow).where(SnapshotRow.entry_id == entry_id).values(extraction_attempts=1)
    )
    await session.flush()


# --- the mark (§1) ------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_metered_deposit_marks_its_new_entry_and_others_leave_it_unmarked(
    db_session: AsyncSession,
) -> None:
    metered = await _deposit(db_session, "a metered note")
    unmetered = await _deposit(db_session, "an operator note", "api-user", metered=False)

    assert await _mark(db_session, metered) == AGENT
    assert await _mark(db_session, unmetered) is None
    listed = {p.entry_id: p for p in await list_pending_snapshots_for_catchup(db_session)}
    assert listed[metered].metered_principal == AGENT
    assert listed[unmetered].metered_principal is None


@pytest.mark.asyncio
async def test_an_entry_from_before_the_mark_is_unmetered_until_a_metered_deposit_lands_on_it(
    db_session: AsyncSession,
) -> None:
    """The second case of §1: without it a pre-migration entry escaped both caps for good."""
    uri = "particles://agent/laptop-mcp/journal"
    with _as(AGENT, metered=False):
        entry_id, _, _ = await deposit_text_versioned(
            db_session,
            text="first page",
            uri_r=uri,
            source_type="CONVERSATION",
            mutability=Mutability.APPEND_ONLY,
            deposited_by=AGENT,
        )
    await db_session.flush()
    assert await _mark(db_session, entry_id) is None
    assert await unclaimed_snapshot_count(db_session, AGENT) == (0, None)

    with _as(AGENT):
        await deposit_text_versioned(
            db_session,
            text="first page\nsecond page",
            uri_r=uri,
            source_type="CONVERSATION",
            mutability=Mutability.APPEND_ONLY,
            deposited_by=AGENT,
        )
    await db_session.flush()

    assert await _mark(db_session, entry_id) == AGENT
    # The snapshot already pending when the mark was set counts too.
    assert await unclaimed_snapshot_count(db_session, AGENT) == (2, entry_id)


# --- the cap (§2) -------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_metered_key_is_refused_at_the_cap_and_admitted_once_one_is_claimed(
    db_session: AsyncSession,
) -> None:
    get_config().api.max_pending_extractions_per_principal = 3
    first = await _deposit(db_session, "note 1")
    await _deposit(db_session, "note 2")
    await _deposit(db_session, "note 3")

    with pytest.raises(PendingExtractionsFull) as refused:
        await _deposit(db_session, "note 4")
    assert "api.max_pending_extractions_per_principal is 3" in str(refused.value)
    entries = (await db_session.execute(select(CorpusEntryRow.entry_id))).all()
    assert len(entries) == 3  # nothing stored

    # A snapshot pass 1 claimed and handed back PENDING stops counting.
    await _claim_one(db_session, first)
    await _deposit(db_session, "note 4")
    assert (await unclaimed_snapshot_count(db_session, AGENT))[0] == 3


@pytest.mark.asyncio
async def test_a_deduplicated_deposit_at_the_cap_is_admitted(db_session: AsyncSession) -> None:
    get_config().api.max_pending_extractions_per_principal = 2
    entry_id = await _deposit(db_session, "note 1")
    await _deposit(db_session, "note 2")

    assert await _deposit(db_session, "note 1") == entry_id  # stores nothing new
    assert (await unclaimed_snapshot_count(db_session, AGENT))[0] == 2


@pytest.mark.asyncio
async def test_one_principals_count_is_its_own_and_unmetered_deposits_are_never_refused(
    db_session: AsyncSession,
) -> None:
    get_config().api.max_pending_extractions_per_principal = 1
    await _deposit(db_session, "mine")
    with pytest.raises(PendingExtractionsFull):
        await _deposit(db_session, "mine again")

    await _deposit(db_session, "another key's", OTHER)
    await _deposit(db_session, "an operator's", "api-user", metered=False)
    await _deposit(db_session, "an operator's second", "api-user", metered=False)


@pytest.mark.asyncio
async def test_zero_turns_the_cap_off(db_session: AsyncSession) -> None:
    get_config().api.max_pending_extractions_per_principal = 0
    for i in range(5):
        await _deposit(db_session, f"note {i}")
    assert (await unclaimed_snapshot_count(db_session, AGENT))[0] == 5


@pytest.mark.asyncio
async def test_the_refusal_names_the_entry_stuck_behind_a_handed_back_append_only_head(
    db_session: AsyncSession,
) -> None:
    """The one lockout §2 states: fresh snapshots wait behind a head pass 1 keeps handing back."""
    get_config().api.max_pending_extractions_per_principal = 3
    uri = "particles://agent/laptop-mcp/log"
    text = "line 0"
    with _as(AGENT):
        stuck, _, _ = await deposit_text_versioned(
            db_session,
            text=text,
            uri_r=uri,
            source_type="CONVERSATION",
            mutability=Mutability.APPEND_ONLY,
            deposited_by=AGENT,
        )
    await db_session.flush()
    await _claim_one(db_session, stuck)  # the head, handed back
    for i in range(1, 3):
        text += f"\nline {i}"
        with _as(AGENT):
            await deposit_text_versioned(
                db_session,
                text=text,
                uri_r=uri,
                source_type="CONVERSATION",
                mutability=Mutability.APPEND_ONLY,
                deposited_by=AGENT,
            )
        await db_session.flush()
    await _deposit(db_session, "a separate note")

    with pytest.raises(PendingExtractionsFull) as refused:
        await _deposit(db_session, "one more")
    assert refused.value.stuck_entry_id == stuck
    assert f"particles extract {stuck}" in str(refused.value)


def test_two_deposits_one_below_the_cap_admit_exactly_one(cli_db: Path) -> None:
    """The count runs inside the write unit, under the store's writer lock (§2)."""
    from particles.db import get_engine, session_scope

    get_config().api.max_pending_extractions_per_principal = 2

    async def deposit(text: str) -> str:
        async with session_scope() as session:
            with _as(AGENT), writes_committed_under_lock():
                entry_id, _ = await deposit_text(session, text, AGENT, "CONVERSATION", [])
            return entry_id

    async def go() -> list[object]:
        try:
            await deposit("first")
            return list(
                await asyncio.gather(deposit("second"), deposit("third"), return_exceptions=True)
            )
        finally:
            await get_engine().dispose()

    outcomes = asyncio.run(go())
    assert sum(isinstance(o, PendingExtractionsFull) for o in outcomes) == 1
    assert sum(isinstance(o, str) for o in outcomes) == 1


@pytest.mark.asyncio
async def test_a_follow_refused_at_the_cap_is_reported_as_refused(
    db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The primary deposit stands; the follow is named, not logged as a failed fetch."""
    from particles.corpus import deposit as dep

    get_config().api.max_pending_extractions_per_principal = 1
    with _as(AGENT):
        primary, primary_snap = await deposit_text(
            db_session, "a post linking out", AGENT, "CONVERSATION", []
        )
    target = "https://example.org/target"

    async def nested(session: AsyncSession, url: str, **kwargs: Any) -> tuple[str, str]:
        return await deposit_text(session, "the linked page", AGENT, "CONVERSATION", [])

    monkeypatch.setattr(dep, "deposit_url", nested)
    monkeypatch.setattr(dep.egress, "allows_url", lambda _u: True)

    class _Importer:
        DEFAULT_FOLLOW_POST_LINKS = True

        @staticmethod
        def primary_url(_content: bytes) -> str:
            return target

    refused: list[str] = []
    with _as(AGENT):
        await dep._maybe_follow_primary_url(
            db_session,
            _Importer(),
            entry_id=primary,
            snapshot_id=primary_snap,
            deposited_by=AGENT,
            tags=[],
            follow_post_links=True,
            out_follow_refused=refused,
        )
    assert refused == [target]


# --- the routes ---------------------------------------------------------------


def _text(engine: TestClient, text: str, token: str | None = None) -> Any:
    return engine.post(
        "/corpus/deposit/text",
        json={"text": text, "source_type": "CONVERSATION"},
        headers=_bearer(token) if token else {},
    )


def test_the_routes_answer_429_at_the_cap_and_operator_and_harvester_keys_pass(
    engine: TestClient,
) -> None:
    get_config().api.max_pending_extractions_per_principal = 1
    agent = _mint("agent", AGENT)
    hook = _mint("agent", HOOK, reserved_ok=True, **HOOK_ROOTS)

    assert _text(engine, "the agent's first", agent).status_code == 200
    refused = _text(engine, "the agent's second", agent)
    assert refused.status_code == 429, refused.text
    assert "api.max_pending_extractions_per_principal" in refused.json()["detail"]
    agent_route = engine.post(
        "/particles/deposit", json={"text": "via the agent route"}, headers=_bearer(agent)
    )
    assert agent_route.status_code == 429, agent_route.text

    for i in range(3):
        assert _text(engine, f"the operator's {i}").status_code == 200
        assert _text(engine, f"the harvester's {i}", hook).status_code == 200


def _belief(engine: TestClient, token: str, excerpt: str, supersedes: str | None = None) -> Any:
    body: dict[str, Any] = {
        "content": f"A belief read from {excerpt}.",
        "subject_names": ["release train"],
        "confidence": 0.8,
        "source_excerpt": excerpt,
    }
    if supersedes is not None:
        body["supersedes_id"] = supersedes
    route = "/particles/supersede" if supersedes is not None else "/particles/assert"
    return engine.post(route, json=body, headers=_bearer(token))


def _unmarked_entries_of(principal: str) -> list[str]:
    """The entries ``principal`` deposited that carry no metering mark."""

    async def _go() -> list[str]:
        from particles.db import get_engine, session_scope

        try:
            async with session_scope() as s:
                rows = await s.execute(
                    select(CorpusEntryRow.entry_id).where(
                        CorpusEntryRow.deposited_by == principal,
                        CorpusEntryRow.metered_principal.is_(None),
                    )
                )
                return [str(r) for r in rows.scalars()]
        finally:
            await get_engine().dispose()

    return asyncio.run(_go())


def test_an_assert_or_supersede_that_deposits_its_excerpt_is_metered_and_refused_at_the_cap(
    engine: TestClient,
) -> None:
    """Security review of PR 1479, S1: the excerpt deposit is no way around the cap.

    Safe property: every snapshot an agent key's request makes ``PENDING`` is
    counted, or the request is refused with the cap's ``429``.
    """
    get_config().api.max_pending_extractions_per_principal = 2
    agent = _mint("agent", AGENT)

    asserted = _belief(engine, agent, "the first excerpt")
    assert asserted.status_code == 200, asserted.text
    superseded = _belief(
        engine, agent, "the second excerpt", supersedes=asserted.json()["asserted_particle_id"]
    )
    assert superseded.status_code == 200, superseded.text
    assert _unmarked_entries_of(AGENT) == []

    for refused in (
        _belief(engine, agent, "a third excerpt"),
        _belief(
            engine, agent, "a fourth excerpt", supersedes=superseded.json()["asserted_particle_id"]
        ),
    ):
        assert refused.status_code == 429, refused.text
        assert "api.max_pending_extractions_per_principal" in refused.json()["detail"]


def test_every_agent_key_deposit_route_marks_its_entry(engine: TestClient) -> None:
    """Every route an agent key can deposit through leaves the key's mark on the entry."""
    agent = _mint("agent", AGENT)
    calls = [
        _text(engine, "through the text route", agent),
        engine.post(
            "/particles/deposit", json={"text": "through the agent route"}, headers=_bearer(agent)
        ),
        _belief(engine, agent, "through an assertion"),
    ]
    assert [c.status_code for c in calls] == [200, 200, 200], [c.text for c in calls]
    assert _unmarked_entries_of(AGENT) == []


def test_every_routes_gate_binds_the_deposit_meter_to_an_agent_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The route walk: whatever route an agent key calls, its deposits are metered.

    Metering is bound where the principal is, at the gate (S1's class fix), so
    no handler or operation can reach the deposit seam unmetered. The walk
    fails on a route whose gate leaves the meter unbound for an agent key, and
    on a route with no gate this module knows.
    """
    from fastapi.routing import APIRoute

    from particles.api import _keys, auth
    from particles.api.app import app
    from particles.corpus import metering

    agent = auth.Principal(id=AGENT, role=_keys.Role.AGENT, key_id="k-agent")
    operator = auth.Principal(id="api-user", role=_keys.Role.OPERATOR, key_id="k-op")
    capture = auth.Principal(id="codex-hook", role=_keys.Role.CODEX_CAPTURE, key_id="k-capture")
    monkeypatch.setattr(auth, "engine_is_keyless", lambda: False)
    for verify in ("_verify_read_key", "_verify_key", "_verify_write_key", "_verify_operator_key"):
        monkeypatch.setattr(auth, verify, lambda _request, _creds: current["principal"])
    current: dict[str, Any] = {}

    async def _metered_after(gate: Any) -> str | None:
        metering._metered_principal.set("stale")
        await gate(None, None)
        return metering._metered_principal.get()

    gates: set[Any] = set()
    for route in app.routes:
        if not isinstance(route, APIRoute):
            continue
        found = [d.call for d in route.dependant.dependencies if d.call in auth.GATE_DEPENDENCIES]
        if not found:
            # The app shell and /health: reads with no principal to meter.
            assert {(m, route.path) for m in route.methods} <= OPEN, (
                f"{route.path} names no gate, so nothing binds its deposit meter"
            )
            continue
        gates.update(found)
    assert gates
    for gate in gates:
        if gate is auth.CODEX_PROOF_GATE.dependency:
            admitted = [(capture, None)]
        else:
            admitted = [(agent, AGENT), (operator, None)]
            if gate is auth.TEXT_DEPOSIT_GATE.dependency:
                admitted.append((capture, None))
        for principal, meter in admitted:
            current["principal"] = principal
            assert asyncio.run(_metered_after(gate)) == meter, gate.__name__


# --- the steady state (acceptance criteria) -----------------------------------


async def _simulate(session: AsyncSession, per_day: int, days: int) -> list[tuple[int, int, int]]:
    """``(admitted, refused, count after the night)`` per day, under the default settings."""
    days_out: list[tuple[int, int, int]] = []
    for day in range(1, days + 1):
        admitted = refused = 0
        for i in range(per_day):
            try:
                await _deposit(session, f"day {day} note {i}")
                admitted += 1
            except PendingExtractionsFull:
                refused += 1
        listing = await list_pending_snapshots_for_catchup(session)
        chosen, _ = consolidation_mod._selection_with_share(listing, batching=True)
        assert len(chosen) == min(5, len(listing))
        await session.execute(
            update(SnapshotRow)
            .where(SnapshotRow.snapshot_id.in_([c.snapshot_id for c in chosen]))
            .values(extraction_status=ExtractionStatus.COMPLETE.value, extraction_attempts=1)
        )
        await session.flush()
        held, _ = await unclaimed_snapshot_count(session, AGENT)
        days_out.append((admitted, refused, held))
    return days_out


@pytest.mark.asyncio
async def test_eight_a_day_reaches_the_cap_on_day_fifteen_and_is_never_locked_out(
    db_session: AsyncSession,
) -> None:
    days = await _simulate(db_session, per_day=8, days=20)

    assert days[13][2] == 42  # the end of night 14
    assert all(refused == 0 for _, refused, _ in days[:15])
    assert all(refused == 3 and admitted == 5 for admitted, refused, _ in days[15:])
    assert max(held for _, _, held in days) <= 50


@pytest.mark.asyncio
async def test_four_a_day_is_never_refused(db_session: AsyncSession) -> None:
    days = await _simulate(db_session, per_day=4, days=20)
    assert all(refused == 0 and held == 0 for _, refused, held in days)
